form Test
    word in_filename
    word out_filename_wav
    real start_point
    real end_point
endform

long_sound = Open long sound file... 'in_filename$'
sound1 = Extract part... 'start_point' 'end_point' no
intensity$ = Get intensity (dB)

if intensity$ == "--undefined--"
    select sound1
else
    select sound1
    Save as WAV file... 'out_filename_wav$'
endif

plus long_sound
Remove

