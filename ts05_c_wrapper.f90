subroutine ts05_field(parmod, ps, x, y, z, bx, by, bz) bind(C, name="ts05_field")
    use iso_c_binding
    implicit none

    real(c_double), intent(in) :: parmod(10)
    real(c_double), intent(in) :: ps
    real(c_double), intent(in) :: x
    real(c_double), intent(in) :: y
    real(c_double), intent(in) :: z
    real(c_double), intent(out) :: bx
    real(c_double), intent(out) :: by
    real(c_double), intent(out) :: bz
    integer(c_int) :: iopt

    iopt = 0
    call T04_s(iopt, parmod, ps, x, y, z, bx, by, bz)
end subroutine ts05_field
