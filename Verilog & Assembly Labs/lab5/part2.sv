`timescale 1ns / 1ns // `timescale time_unit/time_precision

module part2#(parameter Clock_FREQUENCY = 4)(ClockIn, Reset, Speed, CounterValue);
	input logic ClockIn, Reset;
	input logic [1:0] Speed;
	output logic [3:0] CounterValue;
	
	logic EnableDC; 

	RateDivider #(4) u1(ClockIn, Reset, Speed, EnableDC);
	DisplayCounter u2 (ClockIn, Reset, EnableDC, CounterValue);
	
	
endmodule

module DisplayCounter(Clock, Reset, EnableDC, CounterValue);
	
	input logic Clock, Reset, EnableDC;
	output logic [3:0] CounterValue; 
	
	always_ff@(posedge Clock)
	if(Reset)
	CounterValue <= 0; 
	else if (EnableDC)
	CounterValue <= CounterValue + 1;
	
	
endmodule

module RateDivider #(parameter Clock_FREQUENCY = 4)(ClockIn, Reset, Speed, Enable);
	input logic ClockIn, Reset;
	input logic [1:0] Speed;
	logic [1:0] countrate; 
	output logic Enable; 
	//only change speed when one cycle is done 
	logic [$clog2(Clock_FREQUENCY*4):0] counter;
	logic [$clog2(Clock_FREQUENCY*4):0] out;
	
	assign Enable = (out == 'b0)?'1:'0;
	
		
	always_comb
	begin
	case (Speed)
	
		0:counter = 0;		
		1:begin
			countrate = 1;
			counter = Clock_FREQUENCY*countrate-1;
			end
		2:	begin
			countrate = 2;
			counter = Clock_FREQUENCY*countrate-1;
			end
		3:
			begin
			countrate = 4;
			counter = Clock_FREQUENCY*countrate-1;
			end
	endcase		
	end
	
	always_ff@(posedge ClockIn)
	
	if(Reset || Enable)
	
	out <= counter;
	// if the counter reaches the end, reset the counter with the new speed 
	else
	out <= out-1; 




	
	
endmodule


	
