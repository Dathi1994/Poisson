#include "myfunc.H"

#include <AMReX_BCUtil.H>
#include <AMReX_MLMG.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void advance (amrex::MultiFab& phi_solution,
	      amrex::MultiFab& rhs_ptr,
	      amrex::MultiFab& phi_exact,
              const Geometry& geom,
              const BoxArray& grids,
              const DistributionMapping& dmap,
              const Vector<BCRec>& bc)
{
    /*
      We use an MLABecLaplacian operator:

      (ascalar*acoef - bscalar div bcoef grad) phi = RHS

      to solve the Poisson equation: div grad phi = RHS

     */

    // Fill the ghost cells of each grid from the other grids
    // includes periodic domain boundaries
    phi_solution.FillBoundary(geom.periodicity());

    // Fill non-periodic physical boundaries
    FillDomainBoundary(phi_solution, geom, bc);

    // assorment of solver and parallization options and parameters
    // see AMReX_MLLinOp.H for the defaults, accessors, and mutators
    LPInfo info;

    // Implicit solve using MLABecLaplacian class
    MLABecLaplacian mlabec({geom}, {grids}, {dmap}, info);

    // order of stencil
    int linop_maxorder = 2;
    mlabec.setMaxOrder(linop_maxorder);

    // build array of boundary conditions needed by MLABecLaplacian
    // see Src/Boundary/AMReX_LO_BCTYPES.H for supported types
    std::array<LinOpBCType,AMREX_SPACEDIM> bc_lo;
    std::array<LinOpBCType,AMREX_SPACEDIM> bc_hi;

    for (int n = 0; n < phi_solution.nComp(); ++n)
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            // lo-side BCs
            if (bc[n].lo(idim) == BCType::int_dir) {
                bc_lo[idim] = LinOpBCType::Periodic;
            }
            else if (bc[n].lo(idim) == BCType::foextrap) {
                bc_lo[idim] = LinOpBCType::Neumann;
            }
            else if (bc[n].lo(idim) == BCType::ext_dir) {
                bc_lo[idim] = LinOpBCType::Dirichlet;
            }
            else {
                amrex::Abort("Invalid bc_lo");
            }

            // hi-side BCs
            if (bc[n].hi(idim) == BCType::int_dir) {
                bc_hi[idim] = LinOpBCType::Periodic;
            }
            else if (bc[n].hi(idim) == BCType::foextrap) {
                bc_hi[idim] = LinOpBCType::Neumann;
            }
            else if (bc[n].hi(idim) == BCType::ext_dir) {
                bc_hi[idim] = LinOpBCType::Dirichlet;
            }
            else {
                amrex::Abort("Invalid bc_hi");
            }
        }
    }

    // tell the solver what the domain boundary conditions are
    mlabec.setDomainBC(bc_lo, bc_hi);

    // set the boundary conditions
    mlabec.setLevelBC(0, &phi_solution);

    // scaling factors
    Real ascalar = 0.0;
    Real bscalar = -1.0;
    mlabec.setScalars(ascalar, bscalar);

    // Set up coefficient matrices
    MultiFab acoef(grids, dmap, 1, 0);

    // fill in the acoef MultiFab and load this into the solver
    acoef.setVal(1.0);
    mlabec.setACoeffs(0, acoef);

    // bcoef lives on faces so we make an array of face-centered MultiFabs
    // then we will in face_bcoef MultiFabs and load them into the solver.
    std::array<MultiFab,AMREX_SPACEDIM> face_bcoef;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        const BoxArray& ba = amrex::convert(acoef.boxArray(),
                                            IntVect::TheDimensionVector(idim));
        face_bcoef[idim].define(ba, acoef.DistributionMap(), 1, 0);
        face_bcoef[idim].setVal(1.0);
    }
    mlabec.setBCoeffs(0, amrex::GetArrOfConstPtrs(face_bcoef));

    // build an MLMG solver
    MLMG mlmg(mlabec);

    // set solver parameters
    int max_iter = 100;
    mlmg.setMaxIter(max_iter);
    int max_fmg_iter = 0;
    mlmg.setMaxFmgIter(max_fmg_iter);
    int verbose = 2;
    mlmg.setVerbose(verbose);
    int bottom_verbose = 0;
    mlmg.setBottomVerbose(bottom_verbose);

    // relative and absolute tolerances for linear solve
    const Real tol_rel = 1.e-10;
    const Real tol_abs = 0.0;

    // Solve linear system
    mlmg.solve({&phi_solution}, {&rhs_ptr}, tol_rel, tol_abs);
}

