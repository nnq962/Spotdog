/*
INVERSE KINEMATIC SPOT V4.0 - NNQ
*/

#include <iostream>
#include<cmath>
using namespace std;

float inverse_kinematic_branch1(float x, float y)
{
    float a = 2 * 0.25 * x;
    float b = 2 * 0.25 * y;
    float c = pow(0.11, 2) - pow(0.25, 2) - pow(x, 2) - pow(y, 2);
    
    float q1_temp = atan2(y, x) + acos(-c / sqrt(pow(a, 2) + pow(b, 2)));
    float theta1 = atan2(y - 0.25 * sin(q1_temp), x - 0.25 * cos(q1_temp));
    
    if (theta1 > 0)
        theta1 = -2 * M_PI + theta1;
    
    return theta1;
}

float inverse_kinematic_branch2(float x, float y)
{
    float a = 2 * 0.25 * x;
    float b = 2 * 0.25 * y;
    float c = pow(0.11, 2) - pow(0.25, 2) - pow(x, 2) - pow(y, 2);
    float theta2 = atan2(y, x) + acos(-c / sqrt(pow(a, 2) + pow(b, 2)));
    
    x = x - 0.05 * cos(theta2) - 0.05;
    y = y - 0.05 * sin(theta2);

    a = 2 * 0.20 * x;
    b = 2 * 0.20 * y;
    c = pow(0.11, 2) - pow(0.2, 2) - pow(x, 2) - pow(y, 2);
    
    float q1_temp = atan2(y, x) - acos(-c / sqrt(pow(a, 2) + pow(b, 2)));
    float theta3 = atan2(y - 0.20 * sin(q1_temp), x - 0.20 * cos(q1_temp));
    
    return theta3;
}

int main() 
{
    float motor_hip;
    float motor_knee;

    float x = -0.0241;
    float y = -0.2073;
    
    motor_hip = inverse_kinematic_branch1(x, y);
    motor_knee = inverse_kinematic_branch2(x, y);
    
    cout<<"Motor hip:   "<<motor_hip<<" rad"<<endl;
    cout<<"Motor knee:  "<<motor_knee<<" rad"<<endl;
    
    return 0;
}
