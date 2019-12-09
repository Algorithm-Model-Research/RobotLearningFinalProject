####################################################
# Copyright (C) 2019 Sam Pickell
# Last Updated December 9, 2019
# Draw_and_Get_Coord.py
####################################################
#  The following Turtle Graphics implementation was obtained from this tutorial:
#  https://techwithtim.net/tutorials/python-module-walk-throughs/turtle-module/drawing-with-mouse/
#  It has been modified and added to so that it reports the mouse's coordinates when drawing

#  Using Turtle Graphics
import turtle
from turtle import *

#  Global variable that sets the number of datapoints to collect
LOOP_THRESHOLD = 20

#  Create screen and turtle variables
out_screen = Screen()
my_turtle = Turtle("turtle")
my_turtle.speed(-1)

#  Create two lists to store X and Y coordinates
x_coords = []
y_coords = []

#  Draw function
def turtle_draw(x, y):
    my_turtle.ondrag(None)
    my_turtle.setheading(my_turtle.towards(x, y))
    my_turtle.goto(x, y)
    my_turtle.ondrag(turtle_draw)

    #  Ensure 0 is always positive
    if(x == -0.0):
        x = 0.0

    #  Append the x coordinate to the end of the list
    x_coords.append(x)

    #  Ensure 0 is always positive
    if(y == -0.0):
        y = 0.0

    #  Append the y coordinate to the end of the list
    y_coords.append(y)

    #  End drawing session after a certain threshold is reached
    if(len(x_coords) >= LOOP_THRESHOLD):
        turtle.bye()


#  The main function
def main():

    turtle.listen()

    my_turtle.ondrag(turtle_draw)

    out_screen.mainloop()

main()
print('X coordinates: ', len(x_coords))
print('Y coordinates: ', len(y_coords))
