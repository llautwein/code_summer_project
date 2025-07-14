//parsed geo-file

lc = 0.14294613383568447;
Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0, 0, lc};
Point(3) = {1, 0.55, 0, lc};
Point(4) = {0, 0.55, 0, lc};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1,2,3,4};
Plane Surface(1) ={1};
Mesh 2;
Physical Surface(1) ={1};
