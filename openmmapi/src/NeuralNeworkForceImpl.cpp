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
#include <fstream>

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

    ifstream graphFile(owner.getFile());
    string graphText((istreambuf_iterator<char>(graphFile)), istreambuf_iterator<char>());
    TF_Buffer* buffer = TF_NewBufferFromString(graphText.c_str(), graphText.size());
    graph = TF_NewGraph();
    TF_ImportGraphDefOptions* importOptions = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(graph, buffer, importOptions, status);
    if (TF_GetCode(status) != TF_OK)
        throw OpenMMException(string("Error loading TensorFlow graph: ")+TF_Message(status));
    TF_DeleteImportGraphDefOptions(importOptions);
    TF_DeleteBuffer(buffer);

    // Create the TensorFlow Session.

    TF_SessionOptions* sessionOptions = TF_NewSessionOptions();
    session = TF_NewSession(graph, sessionOptions, status);
    if (TF_GetCode(status) != TF_OK)
        throw OpenMMException(string("Error creating TensorFlow session: ")+TF_Message(status));
    TF_DeleteSessionOptions(sessionOptions);

    // Create the kernel.

    kernel = context.getPlatform().createKernel(CalcNeuralNetworkForceKernel::Name(), context);
    kernel.getAs<CalcNeuralNetworkForceKernel>().initialize(context.getSystem(), owner, session, graph);
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
