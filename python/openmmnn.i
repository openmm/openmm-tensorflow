%module openmmnn

%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>

%{
#include "NeuralNetworkForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

/*
 * Convert C++ exceptions to Python exceptions.
*/
%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

%feature("shadow") NNPlugin::NeuralNetworkForce::NeuralNetworkForce %{
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], str):
            this = _openmmnn.new_NeuralNetworkForce(args[0])
        else:
            import tensorflow as tf
            import os
            import tempfile
            graph = args[0]
            if len(args) == 2:
                session = args[1]
                graph = tf.compat.v1.graph_util.convert_variables_to_constants(session, graph.as_graph_def(), ['energy', 'forces'])
            with tempfile.TemporaryDirectory() as dir:
                tf.io.write_graph(graph, dir, 'graph.pb', as_text=False)
                file = os.path.join(dir, 'graph.pb')
                this = _openmmnn.new_NeuralNetworkForce(file)
        try:
            self.this.append(this)
        except Exception:
            self.this = this
%}

namespace NNPlugin {

class NeuralNetworkForce : public OpenMM::Force {
public:
    NeuralNetworkForce(const std::string& file);
    const std::string& getFile() const;
    void setUsesPeriodicBoundaryConditions(bool periodic);
    bool usesPeriodicBoundaryConditions() const;

    /*
     * Add methods for casting a Force to a NeuralNetworkForce.
    */
    %extend {
        static NNPlugin::NeuralNetworkForce& cast(OpenMM::Force& force) {
            return dynamic_cast<NNPlugin::NeuralNetworkForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<NNPlugin::NeuralNetworkForce*>(&force) != NULL);
        }
    }
};

}
