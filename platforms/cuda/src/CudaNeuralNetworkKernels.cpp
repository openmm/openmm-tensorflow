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

#include "CudaNeuralNetworkKernels.h"
#include "CudaNeuralNetworkKernelSources.h"
#include "openmm/internal/ContextImpl.h"

using namespace NNPlugin;
using namespace OpenMM;
using namespace std;
using namespace caffe2;

void CudaCalcNeuralNetworkForceKernel::initialize(const System& system, const NeuralNetworkForce& force, Workspace& workspace, NetDef& predictModel) {
    cu.setAsCurrent();
    this->workspace = &workspace;
    this->predictModel = &predictModel;
    positionsTensor = workspace.CreateBlob("positions")->GetMutable<TensorCPU>();
    positionsTensor->Resize(3*system.getNumParticles());
    forces.initialize<float>(cu, 3*system.getNumParticles(), "forces");
    CUmodule module = cu.createModule(CudaNeuralNetworkKernelSources::neuralNetworkForce);
    addForcesKernel = cu.getKernel(module, "addForces");
}

double CudaCalcNeuralNetworkForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    context.getPositions(positions);
    int numParticles = cu.getNumAtoms();
    positionsFloat.resize(3*numParticles);
    for (int i = 0; i < numParticles; i++) {
        positionsFloat[3*i] = positions[i][0];
        positionsFloat[3*i+1] = positions[i][1];
        positionsFloat[3*i+2] = positions[i][2];
    }
    TensorCPU positionsCPU = TensorCPU({numParticles, 3}, positionsFloat, NULL);
    positionsTensor->ShareData(positionsCPU);
    CAFFE_ENFORCE(workspace->RunNet(predictModel->name()));
    double energy = 0.0;
    if (includeEnergy) {
        TensorCPU tensor = workspace->GetBlob("energy")->Get<TensorCPU>();
        const float* data = tensor.data<float>();
        energy = data[0];
    }
    if (includeForces) {
        TensorCPU tensor = workspace->GetBlob("forces")->Get<TensorCPU>();
        const float* data = tensor.data<float>();
        forces.upload(data);
        int paddedNumAtoms = cu.getPaddedNumAtoms();
        void* args[] = {&forces.getDevicePointer(), &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms};
        cu.executeKernel(addForcesKernel, args, numParticles);
    }
    return energy;}
