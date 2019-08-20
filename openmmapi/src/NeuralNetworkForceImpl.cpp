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

#include "internal/NeuralNetworkForceImpl.h"
#include "NeuralNetworkKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

using namespace NNPlugin;
using namespace OpenMM;
using namespace std;

NeuralNetworkForceImpl::NeuralNetworkForceImpl(const NeuralNetworkForce& owner) : owner(owner), graph(NULL), session(NULL), status(TF_NewStatus()) {
}

NeuralNetworkForceImpl::~NeuralNetworkForceImpl() {
    if (session != NULL) {
        TF_CloseSession(session, status);
        TF_DeleteSession(session, status);
    }
    if (graph != NULL)
        TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
}

void NeuralNetworkForceImpl::initialize(ContextImpl& context) {
    // Load the graph from the file.

    string graphProto = owner.getGraphProto();
    TF_Buffer* buffer = TF_NewBufferFromString(graphProto.c_str(), graphProto.size());
    graph = TF_NewGraph();
    TF_ImportGraphDefOptions* importOptions = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, buffer, importOptions, status);
    if (TF_GetCode(status) != TF_OK)
        throw OpenMMException(string("Error loading TensorFlow graph: ")+TF_Message(status));
    TF_DeleteImportGraphDefOptions(importOptions);
    TF_DeleteBuffer(buffer);

    // Check that the graph contains all the expected elements and that their types
    // are supported.

    TF_Output positions = {TF_GraphOperationByName(graph, "positions"), 0};
    if (positions.oper == NULL)
        throw OpenMMException("NeuralNetworkForce: the graph does not have a 'positions' input");
    TF_DataType positionsType = TF_OperationOutputType(positions);
    if (positionsType != TF_FLOAT && positionsType != TF_DOUBLE)
        throw OpenMMException("NeuralNetworkForce: 'positions' must have type float32 or float64");
    TF_DataType boxType = TF_FLOAT;
    if (owner.usesPeriodicBoundaryConditions()) {
        TF_Output boxvectors = {TF_GraphOperationByName(graph, "boxvectors"), 0};
        if (boxvectors.oper == NULL)
            throw OpenMMException("NeuralNetworkForce: the graph does not have a 'boxvectors' input");
        boxType = TF_OperationOutputType(boxvectors);
        if (boxType != TF_FLOAT && boxType != TF_DOUBLE)
            throw OpenMMException("NeuralNetworkForce: 'boxvectors' must have type float32 or float64");
    }
    TF_Output energy = {TF_GraphOperationByName(graph, "energy"), 0};
    if (energy.oper == NULL)
        throw OpenMMException("NeuralNetworkForce: the graph does not have an 'energy' output");
    TF_DataType energyType = TF_OperationOutputType(energy);
    if (energyType != TF_FLOAT && energyType != TF_DOUBLE)
        throw OpenMMException("NeuralNetworkForce: 'energy' must have type float32 or float64");
    TF_Output forces = {TF_GraphOperationByName(graph, "forces"), 0};
    if (forces.oper == NULL)
        throw OpenMMException("NeuralNetworkForce: the graph does not have a 'forces' output");
    TF_DataType forcesType = TF_OperationOutputType(forces);
    if (forcesType != TF_FLOAT && forcesType != TF_DOUBLE)
        throw OpenMMException("NeuralNetworkForce: 'forces' must have type float32 or float64");

    // Create the TensorFlow Session.

    TF_SessionOptions* sessionOptions = TF_NewSessionOptions();
    session = TF_NewSession(graph, sessionOptions, status);
    if (TF_GetCode(status) != TF_OK)
        throw OpenMMException(string("Error creating TensorFlow session: ")+TF_Message(status));
    TF_DeleteSessionOptions(sessionOptions);

    // Create the kernel.

    kernel = context.getPlatform().createKernel(CalcNeuralNetworkForceKernel::Name(), context);
    kernel.getAs<CalcNeuralNetworkForceKernel>().initialize(context.getSystem(), owner, session, graph,
            positionsType, boxType, energyType, forcesType);
}

double NeuralNetworkForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcNeuralNetworkForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

std::vector<std::string> NeuralNetworkForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcNeuralNetworkForceKernel::Name());
    return names;
}
