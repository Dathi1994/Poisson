#include "myfunc.H"
#include "mykernel.H"

#include <AMReX_BCRec.H>
#include <AMReX_BCUtil.H>

using namespace amrex;
//void actual_init_phi(amrex::Real> const& rhs, amrex::Real> const& phi_e, amerex::Real> const& phi_i, amrex::Geometry const& geom)
void actual_init_phi (amrex::MultiFab& rhs_ptr, amrex::MultiFab& phi_exact, amrex::MultiFab& phi_initial, amrex::Geometry const& geom)
{
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real, AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();

    for (MFIter mfi(rhs_ptr); mfi.isValid(); ++mfi)
    {
	    const Box& vbx = mfi.validbox();
	    Array4<Real> const rhs = rhs_ptr.array(mfi);
	    Array4<Real> const phi_e = phi_exact.array(mfi);
            Array4<Real> const phi_i = phi_initial.array(mfi);
	    amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
			    {
			       actual_init_phi(i, j, k, rhs, phi_e, phi_i, dx, prob_lo);
			    });
     }

     phi_initial.setVal(0.0);

}

 //solution.setvalue(0.0)













///////////////////////////////////////////

    //***for (MFIter mfi(phi_new); mfi.isValid(); ++mfi)
    //{
    //    const Box& vbx = mfi.validbox();
    //    auto const& phiNew = phi_new.array(mfi);
    //    amrex::ParallelFor(vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
    //            {
    //                init_phi(i, j, k, phiNew, dx, prob_lo);
    //            });
   // }
///////////////////////////////////////////

