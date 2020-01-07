/***************************************************************************
 *
 * Author: "Sjors H.W. Scheres"
 * MRC Laboratory of Molecular Biology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 ***************************************************************************/
/***************************************************************************
 *
 * Authors:     Carlos Oscar S. Sorzano (coss@cnb.csic.es)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#include <iostream>
#include <math.h>

#include "src/euler.h"
#include "src/funcs.h"

/* Euler angles --> matrix ------------------------------------------------- */
void Euler_angles2matrix(RFLOAT alpha, RFLOAT beta, RFLOAT gamma,
                         Matrix2D<RFLOAT> &A, bool homogeneous)
{
    RFLOAT ca, sa, cb, sb, cg, sg;
    RFLOAT cc, cs, sc, ss;

    if (homogeneous)
    {
        A.initZeros(4,4);
        MAT_ELEM(A,3,3)=1;
    }
    else
        if (MAT_XSIZE(A) != 3 || MAT_YSIZE(A) != 3)
            A.resize(3, 3);

    alpha = DEG2RAD(alpha);
    beta  = DEG2RAD(beta);
    gamma = DEG2RAD(gamma);

    ca = cos(alpha);
    cb = cos(beta);
    cg = cos(gamma);
    sa = sin(alpha);
    sb = sin(beta);
    sg = sin(gamma);
    cc = cb * ca;
    cs = cb * sa;
    sc = sb * ca;
    ss = sb * sa;

    A(0, 0) =  cg * cc - sg * sa;
    A(0, 1) =  cg * cs + sg * ca;
    A(0, 2) = -cg * sb;
    A(1, 0) = -sg * cc - cg * sa;
    A(1, 1) = -sg * cs + cg * ca;
    A(1, 2) = sg * sb;
    A(2, 0) =  sc;
    A(2, 1) =  ss;
    A(2, 2) = cb;
}

/* Euler direction --------------------------------------------------------- */
void Euler_angles2direction(RFLOAT alpha, RFLOAT beta,
						    Matrix1D<RFLOAT> &v)
{
    RFLOAT ca, sa, cb, sb;
    RFLOAT sc, ss;

    v.resize(3);
    alpha = DEG2RAD(alpha);
    beta  = DEG2RAD(beta);

    ca = cos(alpha);
    cb = cos(beta);
    sa = sin(alpha);
    sb = sin(beta);
    sc = sb * ca;
    ss = sb * sa;

    v(0) = sc;
    v(1) = ss;
    v(2) = cb;
}

long double digammal(long double x)
{
	/* force into the interval 1..3 */
	if( x < 0.0L )
		return digammal(1.0L-x)+M_PIl/tanl(M_PIl*(1.0L-x)) ;	/* reflection formula */
	else if( x < 1.0L )
		return digammal(1.0L+x)-1.0L/x ;
	else if ( x == 1.0L)
		return -M_GAMMAl ;
	else if ( x == 2.0L)
		return 1.0L-M_GAMMAl ;
	else if ( x == 3.0L)
		return 1.5L-M_GAMMAl ;
	else if ( x > 3.0L)
		/* duplication formula */
		return 0.5L*(digammal(x/2.0L)+digammal((x+1.0L)/2.0L))+M_LN2l ;
	else
	{
		/* Just for your information, the following lines contain
		* the Maple source code to re-generate the table that is
		* eventually becoming the Kncoe[] array below
		* interface(prettyprint=0) :
		* Digits := 63 :
		* r := 0 :
		* 
		* for l from 1 to 60 do
		* 	d := binomial(-1/2,l) :
		* 	r := r+d*(-1)^l*(Zeta(2*l+1) -1) ;
		* 	evalf(r) ;
		* 	print(%,evalf(1+Psi(1)-r)) ;
		*o d :
		* 
		* for N from 1 to 28 do
		* 	r := 0 :
		* 	n := N-1 :
		*
 		*	for l from iquo(n+3,2) to 70 do
		*		d := 0 :
 		*		for s from 0 to n+1 do
 		*		 d := d+(-1)^s*binomial(n+1,s)*binomial((s-1)/2,l) :
 		*		od :
 		*		if 2*l-n > 1 then
 		*		r := r+d*(-1)^l*(Zeta(2*l-n) -1) :
 		*		fi :
 		*	od :
 		*	print(evalf((-1)^n*2*r)) ;
 		*od :
 		*quit :
		*/
		static long double Kncoe[] = { .30459198558715155634315638246624251L,
		.72037977439182833573548891941219706L, -.12454959243861367729528855995001087L,
		.27769457331927827002810119567456810e-1L, -.67762371439822456447373550186163070e-2L,
		.17238755142247705209823876688592170e-2L, -.44817699064252933515310345718960928e-3L,
		.11793660000155572716272710617753373e-3L, -.31253894280980134452125172274246963e-4L,
		.83173997012173283398932708991137488e-5L, -.22191427643780045431149221890172210e-5L,
		.59302266729329346291029599913617915e-6L, -.15863051191470655433559920279603632e-6L,
		.42459203983193603241777510648681429e-7L, -.11369129616951114238848106591780146e-7L,
		.304502217295931698401459168423403510e-8L, -.81568455080753152802915013641723686e-9L,
		.21852324749975455125936715817306383e-9L, -.58546491441689515680751900276454407e-10L,
		.15686348450871204869813586459513648e-10L, -.42029496273143231373796179302482033e-11L,
		.11261435719264907097227520956710754e-11L, -.30174353636860279765375177200637590e-12L,
		.80850955256389526647406571868193768e-13L, -.21663779809421233144009565199997351e-13L,
		.58047634271339391495076374966835526e-14L, -.15553767189204733561108869588173845e-14L,
		.41676108598040807753707828039353330e-15L, -.11167065064221317094734023242188463e-15L } ;

		register long double Tn_1 = 1.0L ;	/* T_{n-1}(x), started at n=1 */
		register long double Tn = x-2.0L ;	/* T_{n}(x) , started at n=1 */
		register long double resul = Kncoe[0] + Kncoe[1]*Tn ;

		x -= 2.0L ;

		for(int n = 2 ; n < sizeof(Kncoe)/sizeof(long double) ;n++)
		{
			const long double Tn1 = 2.0L * x * Tn - Tn_1 ;	/* Chebyshev recursion, Eq. 22.7.4 Abramowitz-Stegun */
			resul += Kncoe[n]*Tn1 ;
			Tn_1 = Tn ;
			Tn = Tn1 ;
		}
		return resul ;
	}
}

void Euler_angles2quat(RFLOAT alpha, RFLOAT beta,
                       Matrix1D<RFLOAT> &v)
{
    RFLOAT ca, sa, cb, sb;
    RFLOAT r11, r12, r13,
           r21, r22, r23,
           r31, r32, r33;

    v.resize(4);
    alpha = DEG2RAD(alpha);
    beta  = DEG2RAD(beta);

    ca = cos(alpha);
    cb = cos(beta);
    sa = sin(alpha);
    sb = sin(beta);
    r11 = ca*cb, r12 = -sa, r13 = ca*sb;
    r21 = cb*sa, r22 = ca,  r23 = sa*sb;
    r31 = -sb,   r32 = 0.,  r33 = cb;

    v(0) = 0.25*sqrt((r11+r22+r33+1.)*(r11+r22+r33+1.) + (r32-r23)*(r32-r23)
            + (r13-r31)*(r13-r31)+(r21-r12)*(r21-r12));
    v(1) = 0.25*sqrt((r32-r23)*(r32-r23) + (r11-r22-r33+1.)*(r11-r22-r33+1.)
            + (r21+r12)*(r21+r12)+(r31+r13)*(r31+r13));
    if(r32 < r23) v(1) = -v(1);
    v(2) = 0.25*sqrt((r13-r31)*(r13-r31) + (r21+r12)*(r21+r12) + 
            (r22-r11-r33+1.)*(r22-r11-r33+1.) + (r32+r23)*(r32+r23));
    if(r13 < r31) v(2) = -v(2);
    v(3) = 0.25*sqrt((r21-r12)*(r21-r12) + (r31+r13)*(r31+r13) + (r32+r23)*(r32+r23)
            + (r33-r11-r22+1.)*(r33-r11-r22+1.));
    if(r21 < r12) v(3) = -v(3);

}

/* Euler direction2angles ------------------------------- */
//gamma is useless but I keep it for simmetry
//with Euler_direction
void Euler_direction2angles(Matrix1D<RFLOAT> &v0,
                            RFLOAT &alpha, RFLOAT &beta)
{
	// Aug25,2015 - Shaoda
	// This function can recover tilt (b) as small as 0.0001 degrees
	// It replaces a more complicated version in the code before Aug2015
    Matrix1D<RFLOAT> v;

    // Make sure the vector is normalised
    v.resize(3);
    v = v0;
    v.selfNormalize();

    // Tilt (b) should be [0, +180] degrees. Rot (a) should be [-180, +180] degrees
    alpha = RAD2DEG(atan2(v(1), v(0))); // 'atan2' returns an angle within [-pi, +pi] radians for rot
    beta = RAD2DEG(acos(v(2))); // 'acos' returns an angle within [0, +pi] radians for tilt

    // The following is done to keep in line with the results from old codes
    // If tilt (b) = 0 or 180 degrees, sin(b) = 0, rot (a) cannot be calculated from the direction
    if ( (fabs(beta) < 0.001) || (fabs(beta - 180.) < 0.001) )
    	alpha = 0.;

    return;

}

/* Matrix --> Euler angles ------------------------------------------------- */
#define CHECK
//#define DEBUG_EULER
void Euler_matrix2angles(const Matrix2D<RFLOAT> &A, RFLOAT &alpha,
                         RFLOAT &beta, RFLOAT &gamma)
{
    RFLOAT abs_sb, sign_sb;

    if (MAT_XSIZE(A) != 3 || MAT_YSIZE(A) != 3)
        REPORT_ERROR( "Euler_matrix2angles: The Euler matrix is not 3x3");

    abs_sb = sqrt(A(0, 2) * A(0, 2) + A(1, 2) * A(1, 2));
    if (abs_sb > 16*FLT_EPSILON)
    {
        gamma = atan2(A(1, 2), -A(0, 2));
        alpha = atan2(A(2, 1), A(2, 0));
        if (ABS(sin(gamma)) < FLT_EPSILON)
            sign_sb = SGN(-A(0, 2) / cos(gamma));
        // if (sin(alpha)<FLT_EPSILON) sign_sb=SGN(-A(0,2)/cos(gamma));
        // else sign_sb=(sin(alpha)>0) ? SGN(A(2,1)):-SGN(A(2,1));
        else
            sign_sb = (sin(gamma) > 0) ? SGN(A(1, 2)) : -SGN(A(1, 2));
        beta  = atan2(sign_sb * abs_sb, A(2, 2));
    }
    else
    {
        if (SGN(A(2, 2)) > 0)
        {
            // Let's consider the matrix as a rotation around Z
            alpha = 0;
            beta  = 0;
            gamma = atan2(-A(1, 0), A(0, 0));
        }
        else
        {
            alpha = 0;
            beta  = PI;
            gamma = atan2(A(1, 0), -A(0, 0));
        }
    }

    gamma = RAD2DEG(gamma);
    beta  = RAD2DEG(beta);
    alpha = RAD2DEG(alpha);

#ifdef DEBUG_EULER
    std::cout << "abs_sb " << abs_sb << std::endl;
    std::cout << "A(1,2) " << A(1, 2) << " A(0,2) " << A(0, 2) << " gamma "
    << gamma << std::endl;
    std::cout << "A(2,1) " << A(2, 1) << " A(2,0) " << A(2, 0) << " alpha "
    << alpha << std::endl;
    std::cout << "sign sb " << sign_sb << " A(2,2) " << A(2, 2)
    << " beta " << beta << std::endl;
#endif
}
#undef CHECK

#ifdef NEVERDEFINED
// Michael's method
void Euler_matrix2angles(Matrix2D<RFLOAT> A, RFLOAT *alpha, RFLOAT *beta,
                         RFLOAT *gamma)
{
    RFLOAT abs_sb;

    if (ABS(A(1, 1)) > FLT_EPSILON)
    {
        abs_sb = sqrt((-A(2, 2) * A(1, 2) * A(2, 1) - A(0, 2) * A(2, 0)) / A(1, 1));
    }
    else if (ABS(A(0, 1)) > FLT_EPSILON)
    {
        abs_sb = sqrt((-A(2, 1) * A(2, 2) * A(0, 2) + A(2, 0) * A(1, 2)) / A(0, 1));
    }
    else if (ABS(A(0, 0)) > FLT_EPSILON)
    {
        abs_sb = sqrt((-A(2, 0) * A(2, 2) * A(0, 2) - A(2, 1) * A(1, 2)) / A(0, 0));
    }
    else
        EXIT_ERROR(1, "Don't know how to extract angles");

    if (abs_sb > FLT_EPSILON)
    {
        *beta  = atan2(abs_sb, A(2, 2));
        *alpha = atan2(A(2, 1) / abs_sb, A(2, 0) / abs_sb);
        *gamma = atan2(A(1, 2) / abs_sb, -A(0, 2) / abs_sb);
    }
    else
    {
        *alpha = 0;
        *beta  = 0;
        *gamma = atan2(A(1, 0), A(0, 0));
    }

    *gamma = rad2deg(*gamma);
    *beta  = rad2deg(*beta);
    *alpha = rad2deg(*alpha);
}
#endif
/* Euler up-down correction ------------------------------------------------ */
void Euler_up_down(RFLOAT rot, RFLOAT tilt, RFLOAT psi,
                   RFLOAT &newrot, RFLOAT &newtilt, RFLOAT &newpsi)
{
    newrot  = rot;
    newtilt = tilt + 180;
    newpsi  = -(180 + psi);
}

/* Same view, differently expressed ---------------------------------------- */
void Euler_another_set(RFLOAT rot, RFLOAT tilt, RFLOAT psi,
                       RFLOAT &newrot, RFLOAT &newtilt, RFLOAT &newpsi)
{
    newrot  = rot + 180;
    newtilt = -tilt;
    newpsi  = -180 + psi;
}

/* Euler mirror Y ---------------------------------------------------------- */
void Euler_mirrorY(RFLOAT rot, RFLOAT tilt, RFLOAT psi,
                   RFLOAT &newrot, RFLOAT &newtilt, RFLOAT &newpsi)
{
    newrot  = rot;
    newtilt = tilt + 180;
    newpsi  = -psi;
}

/* Euler mirror X ---------------------------------------------------------- */
void Euler_mirrorX(RFLOAT rot, RFLOAT tilt, RFLOAT psi,
                   RFLOAT &newrot, RFLOAT &newtilt, RFLOAT &newpsi)
{
    newrot  = rot;
    newtilt = tilt + 180;
    newpsi  = 180 - psi;
}

/* Euler mirror XY --------------------------------------------------------- */
void Euler_mirrorXY(RFLOAT rot, RFLOAT tilt, RFLOAT psi,
                    RFLOAT &newrot, RFLOAT &newtilt, RFLOAT &newpsi)
{
    newrot  = rot;
    newtilt = tilt;
    newpsi  = 180 + psi;
}

/* Apply a transformation matrix to Euler angles --------------------------- */
void Euler_apply_transf(const Matrix2D<RFLOAT> &L,
                        const Matrix2D<RFLOAT> &R,
                        RFLOAT rot,
                        RFLOAT tilt,
                        RFLOAT psi,
                        RFLOAT &newrot,
                        RFLOAT &newtilt,
                        RFLOAT &newpsi)
{

    Matrix2D<RFLOAT> euler(3, 3), temp;
    Euler_angles2matrix(rot, tilt, psi, euler);
    temp = L * euler * R;
    Euler_matrix2angles(temp, newrot, newtilt, newpsi);
}

/* Rotate (3D) MultidimArray with 3 Euler angles ------------------------------------- */
void Euler_rotation3DMatrix(RFLOAT rot, RFLOAT tilt, RFLOAT psi, Matrix2D<RFLOAT> &result)
{
    Euler_angles2matrix(rot, tilt, psi, result, true);
}


