# DumpTruckDYNAMIC_NewmarkIntegration
Dynamic analysis of dumping truck mechanism actuated by a massless hydraulic actuator (which imposes a prescribed displacement as a function of time) which evolves in the plane. 

The analysis begins with the kinematics of the unloading process. The links are rigid and neglecting friction in the joints. The center of mass is located at the middle point of the links. 
The prescribed displacement for the actuator is imposed in the form La = f(t). The mechanism in initially at rest.

Develop the coefficients of a 3rd order polynomial to represent the actuator as a function of time, when the initial conditions are given for position and velocities. Parameterize the code.

Gravity: Consider gravity acting on the system (a contibution in the external force vector)

## Absolute Coordinates
Choosing a set of redundant representation (with finite element and rigid body coordinates). Each link is represented with 3 nodes (2 for end nodes and 1 for center of mass). The node in the center of mass is used to obtain the mass matrix etc.
Impose kinematic constraints: assembly constraints, rigid body constraints etc. Finally derive the set of active coordinates.
Develop the equations of motion and linearize.

## Newmark Integration scheme (Implicit)
Use Newmkark Integration for solving nonlinear equations of motion along with damping.

## Output
Finding the evolution in time for bar 5 at position, velocity and acceleration levels as well as the force needed for the actuator to deploy the mechanism

