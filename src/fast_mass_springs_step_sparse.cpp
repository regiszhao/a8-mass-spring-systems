#include "fast_mass_springs_step_sparse.h"
#include <igl/matlab_format.h>

void fast_mass_springs_step_sparse(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXi & b,
  const double delta_t,
  const Eigen::MatrixXd & fext,
  const Eigen::VectorXd & r,
  const Eigen::SparseMatrix<double>  & M,
  const Eigen::SparseMatrix<double>  & A,
  const Eigen::SparseMatrix<double>  & C,
  const Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > & prefactorization,
  const Eigen::MatrixXd & Uprev,
  const Eigen::MatrixXd & Ucur,
  Eigen::MatrixXd & Unext)
{
  //////////////////////////////////////////////////////////////////////////////

    double w = 1e10;

    // y
    Eigen::MatrixXd y_matrix = (1 / pow(delta_t, 2)) * M * (2 * Ucur - Uprev) + fext + (w * C.transpose() * C * V);

    // initialize Unext, b_matrix, and d
    Unext = Ucur; // initialize Unext to Ucur
    Eigen::MatrixXd l;
    Eigen::MatrixXd d;

    for(int iter = 0;iter < 50;iter++)
    {
        // calculate d
        d = (A * Unext).rowwise().normalized();
        d.array().colwise() *= r.array(); // scale by r
        
        // calculate b_matrix
        l = k * A.transpose() * d + y_matrix;

        // solve
        Unext = prefactorization.solve(l);
    }
  //////////////////////////////////////////////////////////////////////////////
}
