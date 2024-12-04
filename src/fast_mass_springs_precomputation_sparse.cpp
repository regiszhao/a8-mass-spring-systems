#include "fast_mass_springs_precomputation_sparse.h"
#include "signed_incidence_matrix_sparse.h"
#include <vector>

bool fast_mass_springs_precomputation_sparse(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXd & m,
  const Eigen::VectorXi & b,
  const double delta_t,
  Eigen::VectorXd & r,
  Eigen::SparseMatrix<double>  & M,
  Eigen::SparseMatrix<double>  & A,
  Eigen::SparseMatrix<double>  & C,
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > & prefactorization)
{
  /////////////////////////////////////////////////////////////////////////////

	int num_vertices = V.rows();

	// calculate M 
	M.resize(num_vertices, num_vertices);
	// initialize triplet list
	std::vector<Eigen::Triplet<double> > m_ijv;
	// loop through masses
	for (int mass = 0; mass < num_vertices; mass++) {
		m_ijv.emplace_back(mass, mass, m(mass));
	}
	M.setFromTriplets(m_ijv.begin(),m_ijv.end());

	// A
	signed_incidence_matrix_sparse(num_vertices, E, A);

	// r
	Eigen::MatrixXd edge_vecs = A * V;
	r = edge_vecs.rowwise().norm();

	// C
	int num_pinned = b.size();
	C.resize(num_pinned, num_vertices);
	// initialize triplet list
	std::vector<Eigen::Triplet<double> > c_ijv;
	// loop through pinned vertices
	for (int pinned = 0; pinned < num_pinned; pinned++) {
		c_ijv.emplace_back(pinned, b(pinned), 1);
	}
	C.setFromTriplets(c_ijv.begin(),c_ijv.end());

	// Q
	double w = 1e10;
	Eigen::SparseMatrix<double> Q = (k * A.transpose() * A) + ((1 / pow(delta_t, 2)) * M) + (w * C.transpose() * C);
	prefactorization.compute(Q);
	return prefactorization.info() != Eigen::NumericalIssue;
}
