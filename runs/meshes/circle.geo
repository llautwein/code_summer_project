//parsed geo-file

lc = 0.01;

// Circle Geometry
Point(5) = {1.5, 1, 0, lc};
Point(6) = {1, 1.5, 0, lc};
Point(7) = {0.5, 1, 0, lc};
Point(8) = {1, 0.5, 0, lc};
Point(9) = {1, 1, 0, lc};

Circle(5) = {5, 9, 6};
Circle(6) = {6, 9, 7};
Circle(7) = {7, 9, 8};
Circle(8) = {8, 9, 5};

Curve Loop(2) = {5,6,7,8};


Plane Surface(2) ={2};
Mesh 2;
Physical Surface(2) ={2};
