writeInfoLine: ""

for i to 150

	selectObject: i
	pitch_object = To Pitch: 0,75,600
	selectObject: pitch_object
	frames = Get number of frames
	f0_accumulator = 0
	
	for j to frames
		f0 = Get value in frame: j,"Hertz"
		
		if f0 == undefined
			f0 = 0
		endif
		
		f0_accumulator = f0_accumulator + f0
	endfor
	
	f0_mean = f0_accumulator/frames
	appendInfoLine: f0_mean
	removeObject: pitch_object
	
	appendInfoLine: ""
endfor
