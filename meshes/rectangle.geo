//parsed geo-file

lc = 0.1;

// Rectangle Geometry
Point(6) = {0, 0, 0, lc};
Point(7) = {0.5, 0, 0, lc};
Point(8) = {0.5, 2, 0, lc};
Point(9) = {0, 2, 0, lc};
Line(5) = {6, 7};
Line(6) = {7, 8};
Line(7) = {8, 9};
Line(8) = {9, 6};
Curve Loop(2) = {1,2,3,4,5,6,7,8};
Plane Surface(2) ={2};
Mesh 2;
Physical Surface(2) ={2};
