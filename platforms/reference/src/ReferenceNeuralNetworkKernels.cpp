/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2018 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "ReferenceNeuralNetworkKernels.h"
#include "NeuralNetworkForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/RealVec.h"
#include "openmm/reference/ReferencePlatform.h"

using namespace NNPlugin;
using namespace OpenMM;
using namespace std;

static vector<RealVec>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->positions);
}

static vector<RealVec>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->forces);
}

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return (Vec3*) data->periodicBoxVectors;
}

ReferenceCalcNeuralNetworkForceKernel::~ReferenceCalcNeuralNetworkForceKernel() {
    if (positionsTensor != NULL)
        TF_DeleteTensor(positionsTensor);
    if (boxVectorsTensor != NULL)
        TF_DeleteTensor(boxVectorsTensor);
}

void ReferenceCalcNeuralNetworkForceKernel::initialize(const System& system, const NeuralNetworkForce& force, TF_Session* session, TF_Graph* graph) {
    this->session = session;
    this->graph = graph;
    usePeriodic = force.usesPeriodicBoundaryConditions();
    int numParticles = system.getNumParticles();
    int64_t positionsDims[] = {numParticles, 3};
    positionsTensor = TF_AllocateTensor(TF_FLOAT, positionsDims, 2, numParticles*3*sizeof(float));
    int64_t boxVectorsDims[] = {3, 3};
    boxVectorsTensor = TF_AllocateTensor(TF_FLOAT, boxVectorsDims, 2, 9*sizeof(float));
}

double ReferenceCalcNeuralNetworkForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<RealVec>& pos = extractPositions(context);
    vector<RealVec>& force = extractForces(context);
    int numParticles = pos.size();
    float* positions = reinterpret_cast<float*>(TF_TensorData(positionsTensor));
    for (int i = 0; i < numParticles; i++) {
        positions[3*i] = pos[i][0];
        positions[3*i+1] = pos[i][1];
        positions[3*i+2] = pos[i][2];
    }
    if (usePeriodic) {
        Vec3* box = extractBoxVectors(context);
        float* boxVectors = reinterpret_cast<float*>(TF_TensorData(boxVectorsTensor));
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                boxVectors[3*i+j] = box[i][j];
    }
    vector<TF_Output> inputs, outputs;
    int forceOutputIndex = 0;
    if (includeEnergy) {
        outputs.push_back({TF_GraphOperationByName(graph, "energy"), 0});
        if (outputs[0].oper == NULL)
            throw OpenMMException("NeuralNetworkForce: the graph does not have an 'energy' output");
    }
    if (includeForces) {
        forceOutputIndex = outputs.size();
        outputs.push_back({TF_GraphOperationByName(graph, "forces"), 0});
        if (outputs[forceOutputIndex].oper == NULL)
            throw OpenMMException("NeuralNetworkForce: the graph does not have a 'forces' output");
    }
    vector<TF_Tensor*> inputTensors, outputTensors(outputs.size());
    inputs.push_back({TF_GraphOperationByName(graph, "positions"), 0});
    if (inputs[0].oper == NULL)
        throw OpenMMException("NeuralNetworkForce: the graph does not have a 'positions' input");
    inputTensors.push_back(positionsTensor);
    if (usePeriodic) {
        inputs.push_back({TF_GraphOperationByName(graph, "boxvectors"), 0});
        if (inputs[1].oper == NULL)
            throw OpenMMException("NeuralNetworkForce: the graph does not have a 'boxvectors' input");
        inputTensors.push_back(boxVectorsTensor);
    }
    TF_Status* status = TF_NewStatus();
    TF_SessionRun(session, NULL, &inputs[0], &inputTensors[0], inputs.size(),
                  &outputs[0], &outputTensors[0], outputs.size(),
                  NULL, 0, NULL, status);
    if (TF_GetCode(status) != TF_OK)
        throw OpenMMException(string("Error running TensorFlow session: ")+TF_Message(status));
    TF_DeleteStatus(status);
    double energy = 0.0;
    if (includeEnergy) {
        const float* data = reinterpret_cast<float*>(TF_TensorData(outputTensors[0]));
        energy = data[0];
    }
    if (includeForces) {
        const float* data = reinterpret_cast<float*>(TF_TensorData(outputTensors[forceOutputIndex]));
        for (int i = 0; i < numParticles; i++) {
            force[i][0] += data[3*i];
            force[i][1] += data[3*i+1];
            force[i][2] += data[3*i+2];
        }
    }
    return energy;
}
