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

#include "OpenCLNeuralNetworkKernels.h"
#include "OpenCLNeuralNetworkKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include <map>

using namespace NNPlugin;
using namespace OpenMM;
using namespace std;

OpenCLCalcNeuralNetworkForceKernel::~OpenCLCalcNeuralNetworkForceKernel() {
    if (positionsTensor != NULL)
        TF_DeleteTensor(positionsTensor);
    if (boxVectorsTensor != NULL)
        TF_DeleteTensor(boxVectorsTensor);
}

void OpenCLCalcNeuralNetworkForceKernel::initialize(const System& system, const NeuralNetworkForce& force, TF_Session* session, TF_Graph* graph,
            TF_DataType positionsType, TF_DataType boxType, TF_DataType energyType, TF_DataType forcesType) {
    this->session = session;
    this->graph = graph;
    this->positionsType = positionsType;
    this->boxType = boxType;
    this->energyType = energyType;
    this->forcesType = forcesType;
    usePeriodic = force.usesPeriodicBoundaryConditions();
    int numParticles = system.getNumParticles();

    // Construct input tensors.

    int64_t positionsDims[] = {numParticles, 3};
    positionsTensor = TF_AllocateTensor(positionsType, positionsDims, 2, numParticles*3*TF_DataTypeSize(positionsType));
    if (usePeriodic) {
        int64_t boxVectorsDims[] = {3, 3};
        boxVectorsTensor = TF_AllocateTensor(boxType, boxVectorsDims, 2, 9*TF_DataTypeSize(boxType));
    }

    // Inititalize OpenCL objects.

    networkForces.initialize(cl, 3*numParticles, TF_DataTypeSize(forcesType), "networkForces");
    map<string, string> defines;
    if (forcesType == TF_FLOAT)
        defines["FORCES_TYPE"] = "float";
    else
        defines["FORCES_TYPE"] = "double";
    cl::Program program = cl.createProgram(OpenCLNeuralNetworkKernelSources::neuralNetworkForce, defines);
    addForcesKernel = cl::Kernel(program, "addForces");
}

double OpenCLCalcNeuralNetworkForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3> pos;
    context.getPositions(pos);
    int numParticles = cl.getNumAtoms();
    if (positionsType == TF_FLOAT) {
        float* positions = reinterpret_cast<float*>(TF_TensorData(positionsTensor));
        for (int i = 0; i < numParticles; i++) {
            positions[3*i] = pos[i][0];
            positions[3*i+1] = pos[i][1];
            positions[3*i+2] = pos[i][2];
        }
    }
    else {
        double* positions = reinterpret_cast<double*>(TF_TensorData(positionsTensor));
        for (int i = 0; i < numParticles; i++) {
            positions[3*i] = pos[i][0];
            positions[3*i+1] = pos[i][1];
            positions[3*i+2] = pos[i][2];
        }
    }
    if (usePeriodic) {
        Vec3 box[3];
        cl.getPeriodicBoxVectors(box[0], box[1], box[2]);
        if (boxType == TF_FLOAT) {
            float* boxVectors = reinterpret_cast<float*>(TF_TensorData(boxVectorsTensor));
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    boxVectors[3*i+j] = box[i][j];
        }
        else {
            double* boxVectors = reinterpret_cast<double*>(TF_TensorData(boxVectorsTensor));
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    boxVectors[3*i+j] = box[i][j];
        }
    }
    vector<TF_Output> inputs, outputs;
    int forceOutputIndex = 0;
    if (includeEnergy)
        outputs.push_back({TF_GraphOperationByName(graph, "energy"), 0});
    if (includeForces) {
        forceOutputIndex = outputs.size();
        outputs.push_back({TF_GraphOperationByName(graph, "forces"), 0});
    }
    vector<TF_Tensor*> inputTensors, outputTensors(outputs.size());
    inputs.push_back({TF_GraphOperationByName(graph, "positions"), 0});
    inputTensors.push_back(positionsTensor);
    if (usePeriodic) {
        inputs.push_back({TF_GraphOperationByName(graph, "boxvectors"), 0});
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
        if (energyType == TF_FLOAT)
            energy = reinterpret_cast<float*>(TF_TensorData(outputTensors[0]))[0];
        else
            energy = reinterpret_cast<double*>(TF_TensorData(outputTensors[0]))[0];
    }
    if (includeForces) {
        const void* data = TF_TensorData(outputTensors[forceOutputIndex]);
        networkForces.upload(data);
        addForcesKernel.setArg<cl::Buffer>(0, networkForces.getDeviceBuffer());
        addForcesKernel.setArg<cl::Buffer>(1, cl.getForceBuffers().getDeviceBuffer());
        addForcesKernel.setArg<cl::Buffer>(2, cl.getAtomIndexArray().getDeviceBuffer());
        addForcesKernel.setArg<cl_int>(3, numParticles);
        cl.executeKernel(addForcesKernel, numParticles);
    }
    return energy;
}

