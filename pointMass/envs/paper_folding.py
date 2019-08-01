
# piece of paper
thickness_of_paper = 0.0001 #m
width_of_paper = 0.21 #mm 

def fold(n, thickness_of_paper, width_of_paper):
	thickness_of_paper = thickness_of_paper*2
	width_of_paper = width_of_paper/2
	print('#', n, " Thickness:", thickness_of_paper, "Width", width_of_paper)
	return thickness_of_paper, width_of_paper


number_of_folds = 42

for n in range(0, number_of_folds):
	thickness_of_paper, width_of_paper = fold(n, thickness_of_paper, width_of_paper)

moon = 385000000
print(moon-thickness_of_paper)