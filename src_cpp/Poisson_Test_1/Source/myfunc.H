#ifndef MYFUNC_H_
#define MYFUNC_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>
#include <AMReX_LO_BCTYPES.H>

void main_main ();

void advance (amrex::MultiFab& phi_solution,
	      amrex::MultiFab& phi_exact,
	      amrex::MultiFab& rhs_ptr,
              const amrex::Geometry& geom,
              const amrex::BoxArray& grids,
              const amrex::DistributionMapping& dmap,
              const amrex::Vector<amrex::BCRec>& bc);

void actual_init_phi (amrex::MultiFab& phi_solution, 
		      amrex::MultiFab& phi_exact,
	              amrex::MultiFab& rhs_ptr,
		      amrex::Geometry const& geom);



#endif
