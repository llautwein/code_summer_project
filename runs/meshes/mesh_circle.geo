//parsed geo-file

lc = 0.01;

// Circle Geometry
Point(1) = {0.5, 0.0, 0, lc};
Point(2) = {0.0, 0.5, 0, lc};
Point(3) = {-0.5, 0.0, 0, lc};
Point(4) = {0.0, -0.5, 0, lc};
Point(5) = {0.0, 0.0, 0, lc};

Circle(1) = {1, 5, 2};
Circle(2) = {2, 5, 3};
Circle(3) = {3, 5, 4};
Circle(4) = {4, 5, 1};

Curve Loop(1) = {1,2,3,4};


Plane Surface(1) ={1};
Mesh 2;
Physical Surface(1) ={1};
