/* 
ABOUT:
	This program uses an iterative implicit method of running the Crank-Nicolson
	method in parallel. The iterative nature of this method allows it to be 
	applied to nonlinear as well as linear problems. Here we apply it to the 
	(linear) Schrodinger equation and the (nonlinear) Gross-Pitaevskii 
	eqauation, but it could used for other parabolic PDEs. 
	
	The Crank-Nicolson method is implicit, so a matrix inversion problem must be
	solved each timestep rather than the usual looping through the positions you
	see in the Euler method, for example. The advantage is more stabilitiy.

COMPILE:
	g++ -fopenmp iterative_parallel_CN.cpp -std=c++11 -lm -Wall -O3 -o iterative_parallel_CN.exe

INPUTS:
	See initialization of variables, or the iterative_parallel_CN.inp file

OUTPUTS:
	- A summary file, containing the norm, average location, and variance over time
	- Raw output files containing the probablility amplitude and imaginary and real
	  components wavefunction at every spatial point after every "noutputstep"
	  number of time steps. These can be used to create an animation.
*/

#include <cstdlib>  // Standard stuff (exit, malloc, EXIT_FAILURE, atoi, etc)
#include <iostream> // Input/output to stdout
#include <fstream>  // Input/output to files
#include <cmath>    // Math fuctions
#include <vector>   // Vectors
#include <iomanip>  // Extra input/output options (setw, etc)
#include <string>   // Introduces a type which can hold strings
#include <complex>  // Complex numbers and functions
#include <random>   // Random nuber generation
#include <chrono>   // Timekeeping to quantify how long it takes to run parts
#include "omp.h"    // OpenMP, for parallel computation across processors
#include <dirent.h> // Managing directories etc.
#include <iterator>

using namespace std;
double HBAR = 1;
double MASS = 1;
double PI = 3.14159265358979;
complex<double> I(0, 1);
#define nfnamesize 300

//  Solve a tridiagonal matrix system (LHS-[matrix])*(psinew-[vector]) = ((RHS-[matrix])*(psiold-[vector])) <- Evaluate RHS to RHS-[vector] and solve
//  This is the tridag.c function from simple numerical recipes routine
//  (modified for complex variables)
int tridag(
	int idomain, // Which parallel domain are we in? (ID: 0-ndomains)
	complex<double>* adiag, // The diagonal strip of the LHS matrix
	complex<double>* alower, // The lower diagonal strip of the LHS matrix
	complex<double>* aupper, // The upper diagonal strip of the LHS matrix
	complex<double>* rhs, // The RHS vector (formed out of the product of a tridiagonal RHS matrix with a current known state psiold)
	complex<double>* psinew, // This is where the solution is stored
	int* nsubpoints, // The number of points in each domain (excluding boudaries)
	int* nallpointsleftof // The number of points left of each domain (excluding boudaries)
	)
{
	int jpoint, ijindex, ijlittleindex;
	complex<double> bet, * gam = NULL;
	ijindex = 1 + nallpointsleftof[idomain] - 1;

	/* if the following happens rethink your equations */
	if (abs(adiag[ijindex]) == 0.0)
	{
		cerr << "Error: rethink your equations.\n";
		exit(EXIT_FAILURE);
	}
	/* one vector of workspace is needed */
	if (!(gam = new complex<double>[nsubpoints[idomain]]))
	{
		cerr << "Error: not enough memory to solve matricies.\n";
		exit(EXIT_FAILURE);
	}
	bet = adiag[ijindex];   // bet is used as workspace
	psinew[ijindex] = rhs[ijindex] / bet;

	for (jpoint = 2; jpoint <= nsubpoints[idomain]; jpoint++)
	{
		ijindex = jpoint + nallpointsleftof[idomain] - 1;
		ijlittleindex = jpoint - 1;
		gam[ijlittleindex] = aupper[ijindex - 1] / bet;
		bet = adiag[ijindex] - alower[ijindex] * gam[ijlittleindex];
		if (abs(bet) == 0.0)
		{
			cerr << "Error: Pivoting error in tridag\n";
			exit(EXIT_FAILURE);   // Pivoting Error
		}
		psinew[ijindex] = (rhs[ijindex] - alower[ijindex] * psinew[ijindex - 1]) / bet;
	}

	for (jpoint = nsubpoints[idomain] - 1; jpoint >= 1; jpoint--)
	{
		ijindex = jpoint + nallpointsleftof[idomain] - 1;
		ijlittleindex = jpoint - 1;
		psinew[ijindex] -= gam[ijlittleindex + 1] * psinew[ijindex + 1];  // Backsubstition
	}

	delete[] gam;

	return 0;
}


// Initialise the tridiagonal LHS matrix which will multiply psinew in the tridag() solver function
void initlhs(
	int idomain, /// Which parallel domain are we in? (ID: 0-ndomains)
	complex <double>* alower, // The lower diagonal strip of the LHS matrix - SET HERE
	complex <double>* aupper, // The upper diagonal strip of the LHS matrix - SET HERE
	complex <double>* adiag, // The diagonal strip of the LHS matrix 		- SET HERE
	int* ileftboundarypoint, // The global index of the left boundary point of each domain
	int* nallpointsleftof, // The number of points left of each domain (excluding boudaries)
	int* nsubpoints, // The number of points in each domain (excluding boudaries)
	double* potential, // Array storing the potential at every point (not just in this domain)
	complex <double> alpha, // Constant (dt * I * HBAR / (pow(dx, 2) * 4 * MASS))
	complex <double> beta, // Constant (I * dt / (2 * HBAR))
	double gstrength, // Nonlinear coupling constant (turns linear Schrodinger equation into nonlinear Gross-Pitaevskii equation when set not equal to zero)
	complex <double>* psinew // Guess at new state. Nonzero gstrength means the future potential depends on the future state, psinew
	)
{
	int ijindex, jpoint, ijpotentialindex;
	// Setting the individual lhs matrix for a domain
	for (jpoint = 1; jpoint <= nsubpoints[idomain]; jpoint++)
	{
		ijindex = jpoint + nallpointsleftof[idomain] - 1;
		ijpotentialindex = jpoint + ileftboundarypoint[idomain] - 1;
		alower[ijindex] = -alpha;
		adiag[ijindex] = 1.0
			+ 2.0 * alpha
			+ beta *
			(
				potential[ijpotentialindex]
				+ gstrength * pow(abs(psinew[ijindex]), 2)
				);
		aupper[ijindex] = -alpha;
	}
}


//  Initialise and evaluate the RHS of the equation (LHS-[matrix])*(psinew-[vector]) = ((RHS-[matrix])*(psiold-[vector])) <- Evaluate RHS to RHS-[vector] here
void initrhs(
	int idomain, // Which parallel domain are we in? (ID: 0-ndomains)
	complex <double>* leftboundaryold, // The value of the wavefunction in the past at the left boundary point of each domain (vector)
	complex <double>* rightboundaryold,// The value of the wavefunction in the past at the right boundary point of each domain (vector)
	complex <double>* leftboundarynew, // The value of the wavefunction in the future at the left boundary point of each domain (vector)
	complex <double>* rightboundarynew, // The value of the wavefunction in the future at the right boundary point of each domain (vector)
	complex <double>* rhs, // This is SET by this function
	int* ileftboundarypoint, // The global index of the left boundary point of this domain
	int* nallpointsleftof, // The number of points left of each domain (excluding boudaries)
	int* nsubpoints, // The number of points in each domain (excluding boudaries)
	double* potential, // Array storing the potential at every point (not just in this domain)
	complex <double> alpha, // Constant (dt * I * HBAR / (pow(dx, 2) * 4 * MASS))
	complex <double> beta, // Constant (I * dt / (2 * HBAR))
	double gstrength, // Nonlinear coupling constant (turns linear Schrodinger equation into nonlinear Gross-Pitaevskii equation when set not equal to zero)
	complex <double>* psiold // Old state. Used in multiplying (RHS-[matrix])*(psiold-[vector]), and for nonlinear gstrength potential
	)
{
	int ijindex, jpoint, ijpotentialindex;
	// Setting the individual rhs vector for a domain
	for (jpoint = 2; jpoint <= (nsubpoints[idomain] - 1); jpoint++)
	{
		ijindex = jpoint + nallpointsleftof[idomain] - 1;
		ijpotentialindex = jpoint + ileftboundarypoint[idomain] - 1;
		rhs[ijindex] = alpha * psiold[ijindex - 1]
			+ (
				1.0
				- 2.0 * alpha
				- beta *
				(
					potential[ijpotentialindex]
					+ gstrength * pow(abs(psiold[ijindex]), 2)
					)
				) * psiold[ijindex]
			+ alpha * psiold[ijindex + 1];
	}

	// Left boundary
	jpoint = 1;
	ijindex = jpoint + nallpointsleftof[idomain] - 1;
	ijpotentialindex = jpoint + ileftboundarypoint[idomain] - 1;
	rhs[ijindex] = alpha * (leftboundaryold[idomain] + leftboundarynew[idomain])
		+ (
			1.0
			- 2.0 * alpha
			- beta *
			(
				potential[ijpotentialindex]
				+ gstrength * pow(abs(psiold[ijindex]), 2)
				)
			) * psiold[ijindex]
		+ alpha * psiold[ijindex + 1];

	// Right boundary
	jpoint = nsubpoints[idomain];
	ijindex = jpoint + nallpointsleftof[idomain] - 1;
	ijpotentialindex = jpoint + ileftboundarypoint[idomain] - 1;
	rhs[ijindex] = alpha * psiold[ijindex - 1]
		+ (
			1.0
			- 2.0 * alpha
			- beta *
			(
				potential[ijpotentialindex]
				+ gstrength * pow(abs(psiold[ijindex]), 2)
				)
			) * psiold[ijindex]
		+ alpha * (rightboundaryold[idomain] + rightboundarynew[idomain]);
}



// This is how info is shared across domains: their boundaries are updated to match the correspoding points in the adjacent domains
// Only one processor will excecute this function, sharing info for all. 
void updateboundaries(
	complex <double>* leftboundary, // The value of the wavefunction (in the past or future depending on input to function) at the left boundary point of each domain (vector)
	complex <double>* rightboundary, // The value of the wavefunction (in the past or future depending on input to function) at the right boundary point of each domain (vector)
	complex <double>* psi, // The value of the wavefunction (in the past or future depending on input to function) at each point in space (vector)
	int* nsubpoints, // The number of points in each domain (excluding boudaries)
	int* nallpointsleftof, // The number of points left of each domain (excluding boudaries)
	int ndomains // The number of parallel domains (ndomains = 1 is original Crank-Nicolson)
	)
{
	int idomain, jpoint, ijindex;
	// Update the left boundaries
	for (idomain = 1; idomain < ndomains; idomain++)
	{
		jpoint = nsubpoints[idomain - 1];
		ijindex = jpoint + nallpointsleftof[idomain - 1] - 1;
		leftboundary[idomain] = psi[ijindex];
	}
	// Update the right boundaries
	for (idomain = 0; idomain < (ndomains - 1); idomain++)
	{
		jpoint =  1;
		ijindex = jpoint + nallpointsleftof[idomain + 1] - 1;
		rightboundary[idomain] = psi[ijindex];
	}
}








////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// MAIN: 


int main()
{
	// Declaration of variables:
	FILE* infile;               // Pointer to the input file
	FILE* summaryfile;          // Pointer to one single summary file
	FILE* rawfile;              // Pointer to raw data files made each output step
	char fname[nfnamesize];    // Array of characters of filename
	int i, k, n, idomain, jpoint, ijindex, ijpotentialindex;  // Looped variables
	int noutputstep;            // No. of calculating steps between outputs  [INPUT]
	int nxpoints, npts;        // No. of position points to calculate over  [INPUT]
	int ntpoints;              // No. of time points to calculate over      [INPUT]
	int nallpoints;            // TOTAL number of points, including overlaps and boundaries
	int ndomains, nds;         // No. of parallelisation domains            [INPUT]
	int ninputs;               // No. inputs (to check enough has been input)
	complex<double>* leftboundaryold;        // Holds the value of the left boundary of a domain
	complex<double>* rightboundaryold;       // Holds the value of the right boundary of a domain 
	complex<double>* leftboundarynew;        // Holds the value of the left boundary of a domain
	complex<double>* rightboundarynew;       // Holds the value of the right boundary of a domain 
	int* nallpointsleftof;     // Finds ALL the points which should be indexed before a new domain
	std::chrono::duration<double>* time_elapsed;
	std::chrono::duration<double>* time_elapsed_total;
	int iswitchpsi;            // For selecting the initial wavefunction    [INPUT]
	int iswitchv;              // For selecting which potential to use      [INPUT]
	int niterations;           // Number of iterations of the method to perform each step [INPUT]
	double xleft;              // Left boundary position                    [INPUT]
	double xright;             // Right boundary position                   [INPUT]
	double dt;                 // Time stepsize                             [INPUT]
	double dx;                 // Position stepsize
	double sigma0;             // The initial width of the gaussian         [INPUT]
	double v0;                 //The potential strength                     [INPUT]
	double omega;              //The characteristic frequency of potential  [INPUT]
	double xavg;               // Average position of the wavefunction
	double variance;           // Calculated variance
	double norm;               // Calculated norm
	double kick;               // Momentum wavenumber of the initial state  [INPUT]
	double x0;                 // Initial central position of wavefunction  [INPUT]
	double xx;
	double gstrength;          // Strength of the non-linear coupling       [INPUT]
	double* potential;         // Array of the potential at each x point    [INPUT]
	complex<double> alpha;     // A constant to simplify the algebra
	complex<double> beta;      // A constant to simplify the algebra
	int* nsubpoints;           // No. of spatial points per parallel section
	int* ileftboundarypoint;     // Array of positions of left boundary points
	complex<double>* aupper;   // Array of elements of upper diagonal LHS matrix
	complex<double>* adiag;    // Array of elements of diagonal LHS matrix
	complex<double>* alower;   // Array of elements of lower diagonal LHS matrix
	complex<double>* psinew;   // Array of new psi state
	complex<double>* psiold;   // Array of old psi state
	complex<double>* rhs;      // Array of the RHS in the matrix equation

	omp_set_dynamic(0);

	///////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////
	//INPUTS:

	infile = fopen("iterative_parallel_CN.inp", "r");
	if (!infile) {
		cerr << "Error opening input file = iterative_parallel_CN.inp\n";
		exit(EXIT_FAILURE);
	}

	// Inputs
	ninputs = 0;
	i = 0;
	ninputs += fscanf(infile, "%lf", &gstrength); i++;
	ninputs += fscanf(infile, "%i", &iswitchpsi); i++;
	ninputs += fscanf(infile, "%lf", &x0); i++;
	ninputs += fscanf(infile, "%lf", &sigma0); i++;
	ninputs += fscanf(infile, "%lf", &kick); i++;
	ninputs += fscanf(infile, "%i", &iswitchv); i++;
	ninputs += fscanf(infile, "%lf", &v0); i++;
	ninputs += fscanf(infile, "%lf", &omega); i++;
	ninputs += fscanf(infile, "%lf", &xleft); i++;
	ninputs += fscanf(infile, "%lf", &xright); i++;
	ninputs += fscanf(infile, "%i", &nxpoints); i++;
	ninputs += fscanf(infile, "%lf", &dt); i++;
	ninputs += fscanf(infile, "%i", &ntpoints); i++;
	ninputs += fscanf(infile, "%i", &noutputstep); i++;
	ninputs += fscanf(infile, "%i", &ndomains); i++;
	ninputs += fscanf(infile, "%i", &niterations); i++;
	ninputs += fclose(infile);
	if (ninputs != i)
	{
		cerr << "ERROR: Wrong number of inputs\n";
		exit(EXIT_FAILURE);
	}


	printf("This program solves the Schrodinger equation in a potential\n");
	cout << "\nInput by hitting return between these:\n";
	cout << "The strength of the non-linear coupling constant-(gstrength):   " << gstrength << "\n";
	cout << "The initial state psi: 0->gaussian 1->sech-------(iswitchpsi):  " << iswitchpsi << "\n";
	cout << "The position of the initial gaussian-------------(x0):          " << x0 << "\n";
	cout << "The width of the initial gaussian or sech--------(sigma0):      " << sigma0 << "\n";
	cout << "The initial momentum kick------------------------(kick):        " << kick << "\n";
	cout << "The potential V: 0->none 1->harmonic-------------(iswitchv):    " << iswitchv << "\n";
	cout << "The potential strength---------------------------(v0):          " << v0 << "\n";
	cout << "The potential's characteristic frequency---------(omega):       " << omega << "\n";
	cout << "The leftmost boundary position-------------------(xleft):       " << xleft << "\n";
	cout << "The rightmost boundary position------------------(xright):      " << xright << "\n";
	cout << "The number of spatial points---------------------(nxpoints):    " << nxpoints << "\n";
	cout << "The timestep size--------------------------------(dt):          " << dt << "\n";
	cout << "The number of temporal points--------------------(ntpoints):    " << ntpoints<< "\n";
	cout << "The number of calculating steps between outputs--(noutputstep): " << noutputstep << "\n";
	cout << "Number of parallel domains to break grid into----(ndomains):    " << ndomains << "\n";
	cout << "Number of iterations to perform each step--------(niterations): " << niterations << "\n";


	//Calculate space stepsize (+1 avoids fencepost error)
	dx = (xright - xleft) / ((double)nxpoints + 1);

	time_elapsed = new std::chrono::duration<double>[ndomains];
	time_elapsed_total = new std::chrono::duration<double>[ndomains];

	cout << "-------------------------------------------------------------------------\n";
	cout << ndomains << " domains\n";



	///////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////
	//INITIALIZE ARRAYS:

	// Get starting timepoint
	auto starttime0 = std::chrono::high_resolution_clock::now();
	auto stoptime0 = std::chrono::high_resolution_clock::now();
	auto starttime1 = std::chrono::high_resolution_clock::now();
	auto stoptime1 = std::chrono::high_resolution_clock::now();

	nsubpoints = new int[ndomains];
	ileftboundarypoint = new int[ndomains];
	nallpointsleftof = new int[ndomains];
	leftboundaryold = new complex<double>[ndomains];
	rightboundaryold = new complex<double>[ndomains];
	leftboundarynew = new complex<double>[ndomains];
	rightboundarynew = new complex<double>[ndomains];

	// Calculate sizes of arrays 
	//    (This is a bit silly, but I initially did it this way because I was
	//     planning to allow for domains to overlap)
	i = 0;
	nds = ndomains;
	npts = nxpoints;

	while (nds > 1)
	{
		nsubpoints[i] = floor(npts / nds);
		npts += -nsubpoints[i];
		nds--;
		i++;
	}
	nsubpoints[i] = npts;
	nallpoints = nxpoints;
	nallpointsleftof[0] = 0;

	for (idomain = 1; idomain < ndomains; idomain++)
	{
		nallpointsleftof[idomain] = nallpointsleftof[idomain - 1] + nsubpoints[idomain - 1];
	}

	// Calculate the positions of their left boundary points:
	ileftboundarypoint[0] = 0;
	for (idomain = 1; idomain < ndomains; idomain++)
	{
		ileftboundarypoint[idomain] = ileftboundarypoint[idomain - 1] + nsubpoints[idomain - 1];
	}
	// Check that things make sense
	if (!(nallpoints == nallpointsleftof[ndomains - 1] + nsubpoints[idomain - 1]))
	{
		cerr << "ERROR: nallpoints != nallpointsleftof[ndomains-1] + nsubpoints[idomain-1]\n";
		cerr << "nallpoints = " << nallpoints << endl;
		cerr << "nallpointsleftof[ndomains-1] = " << nallpointsleftof[ndomains - 1] << endl;
		cerr << "nsubpoints[idomain-1] = " << nsubpoints[idomain - 1] << endl;
		exit(EXIT_FAILURE);
	}


	//Allocate memory to the spatial density arrays
	if (
		!(aupper = new complex<double>[nallpoints])
		|| !(adiag = new complex<double>[nallpoints])
		|| !(alower = new complex<double>[nallpoints])
		|| !(psinew = new complex<double>[nallpoints])
		|| !(psiold = new complex<double>[nallpoints])
		|| !(rhs = new complex<double>[nallpoints])
		|| !(potential = new double[nxpoints + 2])) {
		cerr << "Error: not enough memory to initialise arrays.\n";
	}

	///////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////
	//SET INITIAL CONDITIONS:

	//Initialize constants
	alpha = dt * I * HBAR / (pow(dx, 2) * 4 * MASS);
	beta = I * dt / (2 * HBAR);

	//Initialise the potential
	if (iswitchv == 0)
	{ // No potential
		for (k = 0; k <= nxpoints + 1; k++)
			potential[k] = v0;
	}
	else
	{ // Harmonic potential
		for (k = 0; k <= nxpoints + 1; k++)
			potential[k] = pow(omega, 2) * v0 * pow((xleft + ((double)k) * dx), 2);
	}

	//Initialize the wavefunction
	norm = 0;
	for (idomain = 0; idomain < ndomains; idomain++)
	{
		for (jpoint = 1; jpoint <= nsubpoints[idomain]; jpoint++)
		{
			ijindex = jpoint + nallpointsleftof[idomain] - 1;
			xx = xleft + double(ileftboundarypoint[idomain] + jpoint) * dx;

			// For the wavefunction
			if (iswitchpsi == 0) { // Gaussian
				psinew[ijindex] = exp(-pow(xx - x0, 2) / (2 * pow(sigma0, 2)));
			}
			else { // Sech
				psinew[ijindex] = 1.0 / cosh(PI * (xx - x0) / (3 * sqrt(2) * sigma0));
			}
			// Kick
			psinew[ijindex] = psinew[ijindex] * cos(kick * (xx - x0))
				+ I * psinew[ijindex] * sin(kick * (xx - x0));

			if ((jpoint > 0) || (idomain == 0))
				norm = norm + dx * abs(psinew[ijindex]) * abs(psinew[ijindex]);
		}
	}

	// Normalize the wavefunction
	for (idomain = 0; idomain < ndomains; idomain++)
	{
		for (jpoint = 1; jpoint <= nsubpoints[idomain]; jpoint++)
		{
			ijindex = jpoint + nallpointsleftof[idomain] - 1;
			psinew[ijindex] = psinew[ijindex] / pow(norm, 0.5);
		}
	}

	// Hardwall external boundaries
	leftboundaryold[0] = 0;
	rightboundaryold[ndomains - 1] = 0;
	leftboundarynew[0] = 0;
	rightboundarynew[ndomains - 1] = 0;

	// Set the internal boundaries equal to their corresponding points in psinew
	updateboundaries(leftboundaryold, rightboundaryold, psinew, nsubpoints, nallpointsleftof, ndomains);
	updateboundaries(leftboundarynew, rightboundarynew, psinew, nsubpoints, nallpointsleftof, ndomains);

	// Get timepoint
	stoptime0 = std::chrono::high_resolution_clock::now();
	time_elapsed[ndomains - 1] = (stoptime0 - starttime0);



	///////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////
	//INITIALIZE OUTPUT:

	snprintf(fname, nfnamesize, "./iterative_parallel_CN_%diterations_%06d.dat", niterations, 0);
	rawfile = fopen(fname, "w");
	if (!rawfile)
	{
		printf("Error opening file = %s\n", fname);
		exit(EXIT_FAILURE);
	}
	fprintf(rawfile, "# (       x,           amplitude,     real value,     imaginary value,     potential) , at t = %lf \n", 0.0);
	
	// Reset summary variables
	norm = 0;
	variance = 0;
	xavg = 0;
	for (idomain = 0; idomain < ndomains; idomain++)
	{
		for (jpoint = 1; jpoint <= nsubpoints[idomain]; jpoint++)
		{
			ijindex = jpoint + nallpointsleftof[idomain] - 1;
			ijpotentialindex = jpoint + ileftboundarypoint[idomain] - 1;
			xx = xleft + double(ileftboundarypoint[idomain] + jpoint) * dx;

			norm += dx * abs(psinew[ijindex]) * abs(psinew[ijindex]);
			xavg += dx * xx * abs(psinew[ijindex]) * abs(psinew[ijindex]);
			fprintf(rawfile, "%.30e %.30e %.30e %.30e %.30e\n",
				xx, abs(psinew[ijindex]) * abs(psinew[ijindex]), real(psinew[ijindex]), imag(psinew[ijindex]), potential[ijpotentialindex]);
		}
	}

	for (idomain = 0; idomain < ndomains; idomain++)
	{
		for (jpoint = 1; jpoint <= nsubpoints[idomain]; jpoint++)
		{
			ijindex = jpoint + nallpointsleftof[idomain] - 1;
			ijpotentialindex = jpoint + ileftboundarypoint[idomain] - 1;
			xx = xleft + double(ileftboundarypoint[idomain] + jpoint) * dx;

			variance += dx * pow(xx-xavg, 2) * abs(psinew[ijindex]) * abs(psinew[ijindex]);
		}
	}

	fclose(rawfile);
	


	//Summary file
	summaryfile = fopen("iterative_parallel_CN_summary.dat", "w");
	if (!summaryfile)
	{
		printf("Error opening file = iterative_parallel_CN_summary.dat\n");
		exit(EXIT_FAILURE);
	}
	//Headers for summary file
	fprintf(summaryfile, "#        timestep,      time,           norm,        average location,    stddev \n");
	fprintf(summaryfile,"%15i %15.5lf %15.5lf %15.5lf %15.5lf \n",0,0.0,norm,xavg,pow(variance,0.5));



	///////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////
	//LOOP OVER TIME:

	for (n = 0; n < ntpoints; n++)
	{

		// Set psiold = psinew ready for next step
		for (idomain = 0; idomain < ndomains; idomain++)
		{
			for (jpoint = 1; jpoint <= nsubpoints[idomain]; jpoint++)
			{
				ijindex = jpoint + nallpointsleftof[idomain] - 1;
				psiold[ijindex] = psinew[ijindex];
			}
		}

		///////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////
		//OUTPUT:

		//Print the raw data if we're on an ouput step
		if((n+1) % noutputstep == 0)
		{
			snprintf(fname, nfnamesize, "./iterative_parallel_CN_%diterations_%06d.dat", niterations, int((n + 1) / noutputstep));
			rawfile = fopen(fname, "w");
			if (!rawfile)
			{
				printf("Error opening file = %s\n", fname);
				exit(EXIT_FAILURE);
			}
			fprintf(rawfile, "# (       x,           amplitude,     real value,     imaginary value,     potential) , at t = %lf \n", ((double)n) * dt);
			// Reset summary variables
			norm = 0;
			variance = 0;
			xavg = 0;
			for (idomain = 0; idomain < ndomains; idomain++)
			{
				for (jpoint = 1; jpoint <= nsubpoints[idomain]; jpoint++)
				{
					ijindex = jpoint + nallpointsleftof[idomain] - 1;
					ijpotentialindex = jpoint + ileftboundarypoint[idomain] - 1;
					xx = xleft + double(ileftboundarypoint[idomain] + jpoint) * dx;

					norm += dx * abs(psinew[ijindex]) * abs(psinew[ijindex]);
					xavg += dx * xx * abs(psinew[ijindex]) * abs(psinew[ijindex]);
					fprintf(rawfile, "%.30e %.30e %.30e %.30e %.30e\n",
						xx, abs(psinew[ijindex]) * abs(psinew[ijindex]), real(psinew[ijindex]), imag(psinew[ijindex]), potential[ijpotentialindex]);
				}
			}

			for (idomain = 0; idomain < ndomains; idomain++)
			{
				for (jpoint = 1; jpoint <= nsubpoints[idomain]; jpoint++)
				{
					ijindex = jpoint + nallpointsleftof[idomain] - 1;
					ijpotentialindex = jpoint + ileftboundarypoint[idomain] - 1;
					xx = xleft + double(ileftboundarypoint[idomain] + jpoint) * dx;
					
					variance += dx * pow(xx-xavg, 2) * abs(psinew[ijindex]) * abs(psinew[ijindex]);
				}
			}

			fclose(rawfile);
			fprintf(summaryfile,"%15i %15.5lf %15.5lf %15.5lf %15.5lf \n",n,((double)n)*dt,norm,xavg,pow(variance,0.5));
		}

		// Get timepoint
		starttime1 = std::chrono::high_resolution_clock::now();

		///////////////////////////////////////////////////////////////////////////////
		///////////////////////////////////////////////////////////////////////////////
		//UPDATE:

		int nthreadscheck = 0;
		//#############################################################################
		//#############################################################################
		// START PARALLEL THREADS:
		#pragma omp parallel num_threads(ndomains)
		{
			int ID = omp_get_thread_num();

			// THIS IS THE MEAT OF THE ALGORITHM
			// Domains are updated independently, giving an estimate of the future boundary positions
			// This is repeated, getting a better estimate of the future boundaries each time
			int iteration;
			for (iteration = 0; iteration < niterations; iteration++)
			{
				// Update domains separately, with future boundaries equal to past boundaries
				initlhs(ID, alower, aupper, adiag,
					ileftboundarypoint, nallpointsleftof, nsubpoints, potential, alpha, beta, gstrength, psinew);
				initrhs(ID, leftboundaryold, rightboundaryold, leftboundarynew, rightboundarynew, rhs,
					ileftboundarypoint, nallpointsleftof, nsubpoints, potential, alpha, beta, gstrength, psiold);
				tridag(ID, adiag, alower, aupper, rhs, psinew, nsubpoints, nallpointsleftof);
				// psinew is set in this process, but psiold is unchanged ^

				#pragma omp barrier
				// Pass information from one domain to another to inform new boundaries
				#pragma omp master
				{
					updateboundaries(leftboundarynew, rightboundarynew, psinew, nsubpoints, nallpointsleftof, ndomains);
				}
				#pragma omp barrier
			}


			// Update domains separately one final time, now using the positions of the adjacent nodes as the future boundaries
			// rhs is not updated, so the boundaries are only updated in the future
			initlhs(ID, alower, aupper, adiag,
				ileftboundarypoint, nallpointsleftof, nsubpoints, potential, alpha, beta, gstrength, psinew);
			initrhs(ID, leftboundaryold, rightboundaryold, leftboundarynew, rightboundarynew, rhs,
				ileftboundarypoint, nallpointsleftof, nsubpoints, potential, alpha, beta, gstrength, psiold);
			tridag(ID, adiag, alower, aupper, rhs, psinew, nsubpoints, nallpointsleftof);



			#pragma omp single
			nthreadscheck = omp_get_num_threads();
		} // psinew is set in this process, but psiold is unchanged ^
		// END PARALLEL THREADS
		//#############################################################################
		//#############################################################################

		if (nthreadscheck != ndomains)
		{
			cerr << "ERROR: not enough threads could be allocated\n";
			exit(EXIT_FAILURE);
		}

		// Move boundaries (both new aand old) to match free points
		updateboundaries(leftboundaryold, rightboundaryold, psinew, nsubpoints, nallpointsleftof, ndomains);
		updateboundaries(leftboundarynew, rightboundarynew, psinew, nsubpoints, nallpointsleftof, ndomains);

		// Set psiold = psinew ready for next step (you could probably do this in parallel too actually)
		for (idomain = 0; idomain < ndomains; idomain++)
		{
			for (jpoint = 1; jpoint <= nsubpoints[idomain]; jpoint++)
			{
				ijindex = jpoint + nallpointsleftof[idomain] - 1;
				psiold[ijindex] = psinew[ijindex];
			}
		}

		// Get timepoint
		stoptime1 = std::chrono::high_resolution_clock::now();
		time_elapsed[ndomains - 1] += stoptime1 - starttime1;
	}



	if (ndomains == 2)
		cout << "Boundary is at x = " << xleft + (double(ileftboundarypoint[1]) + 0.5) * dx << endl;



	fclose(summaryfile);
	delete[] psiold;
	delete[] psinew;
	delete[] aupper;
	delete[] adiag;
	delete[] alower;
	delete[] rhs;
	delete[] potential;
	delete[] nsubpoints;
	delete[] ileftboundarypoint;
	delete[] nallpointsleftof;
	delete[] leftboundaryold;
	delete[] rightboundaryold;
	delete[] leftboundarynew;
	delete[] rightboundarynew;

	

	// Get ending timepoint
	stoptime1 = std::chrono::high_resolution_clock::now();
	time_elapsed_total[ndomains - 1] = (stoptime1 - starttime0);

	cout << "nxpoints: " << nxpoints << endl;
	cout << "niterations: " << niterations << endl;
	cout << "ntpoints: " << ntpoints << endl;
	cout << "dt: " << dt << endl;
	cout << "\nTest finished. \n";
	cout << "The test took " << (time_elapsed_total[ndomains - 1]).count() << "s total to run.\n\n";
	cout << "Of which, " << (time_elapsed[ndomains - 1]).count() << "s was spent on relevant calculations.\n\n";


	cout << "Compltete. " << endl;

	delete[] time_elapsed;
	delete[] time_elapsed_total;


	cout << "\n ------------ PROGRAM FINISHED  ------------ \n" << endl;
	exit(EXIT_SUCCESS);
}
