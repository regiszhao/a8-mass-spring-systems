#include "fast_mass_springs_precomputation_dense.h"
#include "signed_incidence_matrix_dense.h"
#include <Eigen/Dense>

bool fast_mass_springs_precomputation_dense(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXd & m,
  const Eigen::VectorXi & b,
  const double delta_t,
  Eigen::VectorXd & r,
  Eigen::MatrixXd & M,
  Eigen::MatrixXd & A,
  Eigen::MatrixXd & C,
  Eigen::LLT<Eigen::MatrixXd> & prefactorization)
{
  /////////////////////////////////////////////////////////////////////////////

	int num_vertices = V.rows();

	// calculate M and A
	M = Eigen::MatrixXd(m.asDiagonal());
	signed_incidence_matrix_dense(num_vertices, E, A);

	// r
	Eigen::MatrixXd edge_vecs = A * V;
	r = edge_vecs.rowwise().norm();

	// C
	int num_pinned = b.size();
	C = Eigen::MatrixXd::Zero(num_pinned, num_vertices);
	for (int pinned = 0; pinned < num_pinned; pinned++) {
		C(pinned, b(pinned)) += 1;
	}

	// Q
	double w = 1e10;
	Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(V.rows(),V.rows());
	Q = (k * A.transpose() * A) + ((1 / pow(delta_t, 2)) * M) + (w * C.transpose() * C);
	prefactorization.compute(Q);
	return prefactorization.info() != Eigen::NumericalIssue;
}
