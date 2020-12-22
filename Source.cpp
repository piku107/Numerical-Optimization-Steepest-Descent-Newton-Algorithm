// Numerical Optimization.cpp : This file contains the 'main' function. Program execution begins and ends there.
// Snehashis Paul
// EXPLAIN VARIABLES HERE****************
// x, y carry the values of x[i], y[i]
// fxy returns the value of the Rosenbrock Function at each x, y
// dfdx, dfdy retrun first derivative of the Rosenbrock Function
// dfdx2, dfdy2, dfdxy, dfdyx return the second derivatives of the Rosenbrock Function
// s1, s2 are gradient of fxy at each value of x and y; S is the mtrix containing [s1 s2]T
// xdS, ydS perform x,y update for steepest descent and xdN,ydN perform x,y update for Newton's Algorithm
// H is the hessian matrix for Newton Algorithm
// invH, invHS, SinvHS are various matrix multiplication functions involving H and S



#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;

// Function for the formula
double formula(double x, double y)
{
    //double xSquare = pow(x, 2);
    double fxy = 100 * (pow(y - x*x, 2)) + pow(1 - x, 2);
    return fxy;
}

// Function for the differentiation with respect to x
double diffX(double x, double y)
{
    double dfdx = 400 * (pow(x, 3)) - 400 * x * y + 2 * x - 2;
    return dfdx;
}

// Function for the differentiation with respect to y
double diffY(double x, double y)

{
    double dfdy = 200 * y - 200 * (pow(x, 2));
    return dfdy;
}

// Function for the double differentiation with respect to x
double diffXX(double x, double y)
{
    double dfdx2 = 1200 * pow(x, 2) - 400 * y + 2;
    return dfdx2;
}

// Function for the double differentiation with respect to y
double diffYY(double x, double y)
{
    double dfdy2 = 200;
    return dfdy2;
}

// Function for the differentiation with respect to x and then y
double diffXY(double x, double y)
{
    double dfdxy = -400 * x;
    return dfdxy;
}

// Function for the differentiation with respect to y and then x
double diffYX(double x, double y)
{
    double dfdyx = -400 * x;
    return dfdyx;
}

// Function for the search direction
double searchDirection1(double x, double y)
{
    double s1 = -diffX(x, y);
    return s1;
}

double searchDirection2(double x, double y)
{
    double s2 = -diffY(x, y);
    return s2;
}

// Function for xdS and ydS
double xDS(double x, double alpha, double s1)
{
    double xd = x + alpha * s1;
    return xd;
}

double yDS(double y, double alpha, double s2)
{
    double yd = y + alpha * s2;
    return yd;
}

// Function for Hessian matrix
MatrixXd hessian(double x, double y)
{
    MatrixXd H(2, 2);
    H(0, 0) = diffXX(x, y);
    H(0, 1) = diffXY(x, y);
    H(1, 0) = diffYX(x, y);
    H(1, 1) = diffYY(x, y);

    return H;
}

// Function to find Hessian Inverse
MatrixXd hessianInverse(MatrixXd H)
{
    MatrixXd invH(2, 2);
    if (H.determinant() != 0)
        invH = H.inverse();
    return invH;
}

// Function to multiply the Hessian inverse and [S1 S2]
MatrixXd mul1(MatrixXd invH, double s1, double s2)
{
    MatrixXd S(2, 1);
    S(0, 0) = s1;
    S(1, 0) = s2;
    MatrixXd invHS = invH * S;
    return invHS;

}

// Function to multiply transpose of [S1 S2], the Hessian inverse and [S1 S2]
MatrixXd mul2(MatrixXd invH, double s1, double s2)
{
    MatrixXd S(2, 1);
    S(0, 0) = s1;
    S(1, 0) = s2;
    MatrixXd SinvHS = S.transpose() * invH * S;
    return SinvHS;
}

// Function for the xdN and ydN
double xDN(double x, double alpha, MatrixXd invHS)
{
    double xdN = x + alpha * invHS(0, 0);
    return xdN;
}

double yDN(double y, double alpha, MatrixXd invHS)
{
    double ydN = y + alpha * invHS(1, 0);
    return ydN;
}

int main()
{
    double alphaInitial = 1.0;
    int N = 6000;
    vector <double> x(N + 1);
    vector <double> y(N + 1);
    x[0] = 1.2;
    y[0] = 1.2;

    // Variables
    double s1 = 0, s2 = 0, xd = 0, yd = 0, fd = 0, alpha, e = 0.000009;
    alpha = alphaInitial;
    double c = 0.1, rho = 0.707;

    MatrixXd S(2, 1);
    MatrixXd H, invH, invHS, SinvHS;

    // Enter your choice for choosing the Optimization method
    int choice;
    cout << "Enter you choice: Enter 1 for Steepest Descent and 2 for Newton's Algorithm" << endl;
    cin >> choice;


    // Steepest Descent Algoritm
    if (choice == 1)
    {
        for (int i = 0; i < N; i++)
        {
            alpha = 1.0;

            cout << "Iteration Number: " << i + 1 << endl;
            s1 = searchDirection1(x[i], y[i]);
            s2 = searchDirection2(x[i], y[i]);

            cout << "S1: " << s1 << endl;
            cout << "S2: " << s2 << endl;

            xd = xDS(x[i], alpha, s1);
            yd = yDS(y[i], alpha, s2);

            fd = formula(xd, yd);
            cout << "Fd = " << fd << endl;

            // Backtracking Line Search Algorithm
            while (fd > (formula(x[i], y[i]) - c * alpha * (pow(s1, 2) + pow(s2, 2))))
            {
        
                alpha = rho * alpha;
                xd = xDS(x[i], alpha, s1);
                yd = yDS(y[i], alpha, s2);
                fd = formula(xd, yd);
                
            }

            cout << "Final alpha: " << alpha << endl;

            x[i + 1] = xDS(x[i], alpha, s1);
            cout << "X: " << xd << endl;
            y[i + 1] = yDS(y[i], alpha, s2);
            cout << "Y: " << yd << endl;
            
            // Exit criteria
            if ((abs(x[i+1] - x[i]) < e) && (abs(y[i+1] - y[i]) < e))
                break;

        }
    }


    // Newton's Algorithm
    if (choice == 2)
    {
        for (int i = 0; i < N; i++)
        {
            alpha = 1.0;

            cout << "Iteration Number: " << i + 1 << endl;
            s1 = searchDirection1(x[i], y[i]);
            s2 = searchDirection2(x[i], y[i]);

            S(0, 0) = s1;
            S(1, 0) = s2;

            cout << "S: " << S << endl;

            H = hessian(x[i], y[i]);
            invH = hessianInverse(H);
            invHS = mul1(invH, s1, s2);
            SinvHS = mul2(invH, s1, s2);

            cout << "invHS: " << invHS << endl;

            xd = xDN(x[i], alpha, invHS);
            yd = yDN(y[i], alpha, invHS);

            cout << "X new: " << xd << "  Y new: " << yd << endl;

            fd = formula(xd, yd);
            cout << "Fd: " << fd << endl;

            // Backtracking Line Search Algorithm
            while (fd > (formula(x[i], y[i]) - c * alpha * SinvHS(0, 0)))
            {
                alpha = rho * alpha;
                xd = xDN(x[i], alpha, invHS);
                yd = yDN(y[i], alpha, invHS);
                fd = formula(xd, yd);
                
            }

            x[i + 1] = xDN(x[i], alpha, invHS);
            y[i + 1] = yDN(y[i], alpha, invHS);
            cout << "x " << xd << endl;
            cout << "y " << yd << endl;

            //Exit criteria
            if ((abs(x[i + 1] - x[i]) < e) && (abs(y[i + 1] - y[i]) < e))
                break;
        }
    }

    else if (choice != 1 && choice != 2)
    {
        cout << "Wrong choice. Please select one of the options" << endl;
    }
}