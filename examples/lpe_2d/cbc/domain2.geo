R = 1.0;
A = 0.15;
h = 0.03;

Point(1) = {0, 0, 0, h};
Point(2) = {R, 0, 0, h};
Point(3) = {0, R, 0, h};
Point(4) = {-R, 0, 0, h};
Point(5) = {0, -R, 0, h};
Point(6) = {A, 0, 0, h};
Point(7) = {0, A, 0, h};
Point(8) = {-A, 0, 0, h};
Point(9) = {0, -A, 0, h};

Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};

Line(5) = {6, 7};
Line(6) = {7, 8};
Line(7) = {8, 9};
Line(8) = {9, 6};

Line Loop(1) = {1, 2, 3, 4};
Line Loop(2) = {-5, -6, -7, -8};

Plane Surface(1) = {1,2};

Mesh.Algorithm = 6;
Mesh.Smoothing = 30;
Mesh.ElementOrder = 2;

Physical Surface(0) = {1};
Physical Line(1) = {1, 2, 3, 4};
Physical Line(2) = {5, 6, 7, 8};
