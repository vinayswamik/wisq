from qiskit.transpiler import PassManager, TranspilerError, TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit import qasm2
from pennylane.ops.op_math import sk_decomposition
from pennylane.tape import QuantumTape
import pennylane as qml
import sys


class PennylaneSK(TransformationPass):

    def __init__(self, epsilon=1e-10) -> None:
        """
        Approximately decompose 1q gates to a discrete basis using Pennylane's implementation of Solovay Kitaev.
        Args:
        epsilon : the permitted error of approximation
        """
        super().__init__()
        self.approx_exp = epsilon

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the ``PennylaneSK`` pass on `dag`.

        Args:
            dag: The input dag.

        Returns:
            Output dag with 1q gates synthesized in the discrete target basis.
        """
        for node in dag.op_nodes():
            if not node.name == "rz":
                continue  # ignore all non-rz qubit gates

            angle = node.op.params[0]

            op = qml.RZ(angle, wires=0)

            if self.approx_exp < 1e-15:
                print(
                    "Warning: small approximation epsilon may cause decomposing into Clifford + T to take an indeterminate amount of time. Consider rerunning with a higher value for --approx_epsilon/-ap."
                )

            ops = sk_decomposition(
                op,
                epsilon=self.approx_exp,
                basis_set=("T", "T*", "H", "S", "S*", "X"),
                max_depth=sys.maxsize,
            )
            decomposed = qasm2.loads(QuantumTape(ops).to_openqasm())
            decomposed.remove_final_measurements()

            approx_dag = circuit_to_dag(decomposed)

            # convert to a dag and replace the gate by the approximation
            dag.substitute_node_with_dag(node, approx_dag)

        return dag
