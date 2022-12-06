# fishBoids

Application can be launched with parameters

First parameter is how many thousands of fishes you want
ex: ./main-gpu 10 -> will launch application with 10k fishes

Next parameters are initial factor values. Cohesion, alignment and then separation
ex: ./main-gpu 10 0.5 1.5 2.0 -> will launch app with 10k fishes, 0.5 cohesion factor, 1.5 alignment factor and 2.0 separation factor

Same applies for main-cpu


FishBoids keys:

Space - pause and start animation
Escape - quit application

Q/A - increment/decrement cohesion factor
W/S - increment/decrement alignment factor
E/D - increment/decrement separation factor
