//parsed geo-file

lc = 0.1;
Point(1) = {0.0, -0.25, 0, lc};

Point(2) = {1.0, -0.25, 0, lc};

Point(3) = {1.0, 0.7500499999999999, 0, lc};

Point(4) = {0.0, 0.7500499999999999, 0, lc};

Line(1) = {1, 2};

Line(2) = {2, 3};

Line(3) = {3, 4};

Line(4) = {4, 1};

Curve Loop(1) = {1,2,3,4};

Plane Surface(1) ={1};

Transfinite Line {3} = 1501;

// --- Mesh Refinement Fields ---
Field[1] = Distance;
Field[1].EdgesList = {3};
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = 0.0006666666666666668;
Field[2].LcMax = 0.1;
Field[2].DistMin = 0.0;
Field[2].DistMax = 0.1;
Field[3] = Min;
Field[3].FieldsList = {2};
Background Field = 3;
Mesh 2;
Physical Surface(1) ={1};
