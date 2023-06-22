#ifndef MY_KERNEL_H_
#define MY_KERNEL_H_

#include <AMReX_FArrayBox.H>

using namespace amrex;

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void actual_init_phi (int i, int j, int k,
               amrex::Array4<amrex::Real> const& rhs_ptr,
               amrex::Array4<amrex::Real> const& phi_exact,
	       amrex::Array4<amrex::Real> const& phi_initial,
               GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
               GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo)
{
    constexpr amrex::Real pi = 3.1415926535897932;
    using amrex::Real;
    Real x = prob_lo[0] + (i+Real(0.5)) * dx[0];
    Real y = prob_lo[1] + (j+Real(0.5)) * dx[1];
    
    phi_exact(i,j,0) = (std::cos(pi*x) * (std::cos(pi*y)));

    rhs_ptr(i,j,0) = 0.0; //-2.0 * pi * pi * (std::cos(pi*x) * (std::cos(pi*y)));

   // phi_initial(i,j,0) = 0;
}
#endif
