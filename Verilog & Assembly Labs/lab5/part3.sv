module part3 #(parameter CLOCK_FREQUENCY = 10)(ClockIn, Reset, Start, Letter, DotDashOut, NewBitOut);
	input logic ClockIn, Reset, Start;
	input logic [2:0] Letter;
	output logic DotDashOut, NewBitOut; 
	
	logic [11:0] morse_code; 
	always_comb
	case(Letter)
	
		0: morse_code = 12'b101110000000;
		1: morse_code = 12'b111010101000; 
		2: morse_code = 12'b111010111010;
		3: morse_code = 12'b111010100000;
		4: morse_code = 12'b100000000000;
		5: morse_code = 12'b101011101000;
		6: morse_code = 12'b111011101000;
		7: morse_code = 12'b101010100000;
	endcase
	
	RateDivider #(10) u1(ClockIn, Reset, Start, morse_code,DotDashOut, NewBitOut); //Enable is basically the NewBitOut
	

	//Shiftregister u2 (ClockIn, Reset, morse_code, NewBitOut, DotDashOut);
	
	
	
endmodule 
/*
module Shiftregister(ClockIn, Reset, morse_code, NewBitOut, DotDashOut);
	input logic ClockIn, Reset;
	input logic [11:0] morse_code; 
	output logic DotDashOut; 
	input logic NewBitOut;
	
	logic [3:0] x; 
	
	always_ff@(posedge ClockIn)
	if(Reset)
	begin
		x <= 11; //set the index to the most significant bit 
		DotDashOut <=0;
	end
	else if(NewBitOut && x >= 0)
	begin
		DotDashOut <= morse_code[x]; 
		x <= x-1; 
	end
	
endmodule 
*/
module RateDivider #(parameter CLOCK_FREQUENCY = 500)(ClockIn, Reset, Start, morse_code, DotDashOut, NewBitOut);
	input logic ClockIn, Reset;
	input logic Start; 
	//only change speed when one cycle is done 
	logic [$clog2(CLOCK_FREQUENCY):0] counter;
	logic [$clog2(CLOCK_FREQUENCY):0] out;
	input logic [11:0] morse_code; 
	output logic DotDashOut; 
	output logic NewBitOut;
	
	logic [3:0] x; 
	logic Enable; 

	assign NewBitOut = (out == 'b0)?'1:'0;
	
		
	assign counter = CLOCK_FREQUENCY*0.5 -1; 
	
	always_comb
	if(NewBitOut)
		DotDashOut = morse_code[x]; 
	else if(Reset)
		DotDashOut = 0;
	


	

	
	always_ff@(posedge ClockIn)
	
	if(Reset)
	begin
		out <= counter +1; 
		x <= 11; //set the index to the most significant bit 
	end
	else if(Start)
	begin
	
	out <= counter;
	x <= 11;
	end
	else if(out == 0 && x > 0)
		begin
		out <= counter;
		x <= x-1;
		end

		
	// if the counter reaches the end, reset the counter with the new speed 
	else if(out <= counter)
	out <= out-1; 
	
	
endmodule
