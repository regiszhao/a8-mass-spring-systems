#include "signed_incidence_matrix_dense.h"

void signed_incidence_matrix_dense(
  const int n,
  const Eigen::MatrixXi & E,
  Eigen::MatrixXd & A)
{
  //////////////////////////////////////////////////////////////////////////////
	
	int num_edges = E.rows();

	// initialize A
	A = Eigen::MatrixXd::Zero(num_edges, n);

	// loop through each edge
	for (int edge = 0; edge < num_edges; edge++) {
		A(edge, E(edge, 0)) += 1;
		A(edge, E(edge, 1)) += -1;
	}

  //////////////////////////////////////////////////////////////////////////////
}
