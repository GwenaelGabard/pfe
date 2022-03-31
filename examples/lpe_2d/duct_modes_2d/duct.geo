L = 2.0;
H = 0.5;
h = 0.01;

Point(1) = {0, 0, 0, h};
Point(2) = {L, 0, 0, h};
Point(3) = {L, H, 0, h};
Point(4) = {0, H, 0, h};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Mesh.Algorithm = 6;
Mesh.Smoothing = 30;
Mesh.ElementOrder = 2;

Physical Surface(0) = {1};
Physical Line(1) = {1};
Physical Line(2) = {2};
Physical Line(3) = {3};
Physical Line(4) = {4};
