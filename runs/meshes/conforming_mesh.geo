//parsed geo-file

lc = 0.1;
lc_overlap = 0.0016666666666666668;

Point(1) = {0.0, -0.25, 0, lc};

Point(2) = {1.0, -0.25, 0, lc};

Point(3) = {1.0, 0.7, 0, lc_overlap};

Point(4) = {0.0, 0.7, 0, lc_overlap};

Point(5) = {0.0, 0.8, 0, lc_overlap};

Point(6) = {1.0, 0.8, 0, lc_overlap};

Point(7) = {1.0, 1.75, 0, lc};

Point(8) = {0.0, 1.75, 0, lc};

Line(1) = {1, 2};

Line(2) = {2, 3};

Line(3) = {3, 4};

Line(4) = {4, 1};

Line(5) = {3, 6};

Line(6) = {4, 5};

Line(7) = {5, 6};

Line(8) = {6, 7};

Line(9) = {7, 8};

Line(10) = {8, 5};

Curve Loop(1) = {1,2,3,4};

Plane Surface(1) ={1};

Physical Surface(1) ={1};

Curve Loop(2) = {7,8,9,10};

Plane Surface(2) ={2};

Physical Surface(2) ={2};

Curve Loop(3) = {-3,5,-7,-6};

Plane Surface(3) ={3};

Physical Surface(3) ={3};

Mesh 2;