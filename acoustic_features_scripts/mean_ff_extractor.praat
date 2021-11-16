writeInfoLine: ""

for i to 125
	selectObject: i
	formant = To Formant (burg): 0,5,5500,0.025,50
	selectObject: formant
	for j to 2
		mean_formant_[j] = Get mean: j,0,0,"Hertz"
	endfor
	appendInfoLine: "'mean_formant_[1]'	'mean_formant_[2]'"
	removeObject: formant
endfor