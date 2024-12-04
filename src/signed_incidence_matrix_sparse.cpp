#include "signed_incidence_matrix_sparse.h"
#include <vector>

void signed_incidence_matrix_sparse(
  const int n,
  const Eigen::MatrixXi & E,
  Eigen::SparseMatrix<double>  & A)
{
  //////////////////////////////////////////////////////////////////////////////

	int num_edges = E.rows();
	A.resize(num_edges,n);

	// initialize triplet list
	std::vector<Eigen::Triplet<double> > ijv;

	// loop through edges
	for (int edge = 0; edge < num_edges; edge++) {
		ijv.emplace_back(edge, E(edge, 0), 1);
		ijv.emplace_back(edge, E(edge, 1), -1);
	}

	A.setFromTriplets(ijv.begin(),ijv.end());
  //////////////////////////////////////////////////////////////////////////////
}
