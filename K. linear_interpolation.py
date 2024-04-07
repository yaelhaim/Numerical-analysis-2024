"""

Function linearInterpolation(table_points, point):
    p = []  // Empty list to store x values
    result = 0  // Variable to store the result
    flag = 1  // Variable to indicate whether interpolation or extrapolation was identified

    For each i in the range of length(table_points):
        Append table_points[i][0] to p  // Add the x value to list p

    For each i in the range of length(p) - 1:
        If i <= point <= i + 1:  // If the point lies between two points in the table
            x1 = table_points[i][0]
            x2 = table_points[i + 1][0]
            y1 = table_points[i][1]
            y2 = table_points[i + 1][1]
            result = (((y1 - y2) / (x1 - x2)) * point) + ((y2 * x1) - (y1 * x2)) / (x1 - x2)
            Print "The interpolated value of the point ", point, " is: ", round(result, 4)
            flag = 0  // Indicates interpolation was identified

    If flag:
        x1 = table_points[0][0]
        x2 = table_points[1][0]
        y1 = table_points[0][1]
        y2 = table_points[1][1]
        m = (y1 - y2) / (x1 - x2)
        result = y1 + m * (point - x1)
        Print "The extrapolated value of the point ", point, " is: ", round(result, 4)


"""


from colors import bcolors

def linearInterpolation(table_points, point):
    length = len(table_points)
    for i in range(length - 1):
        if table_points[i][0] <= point <= table_points[i + 1][0]:
            x1 = table_points[i][0]
            x2 = table_points[i + 1][0]
            y1 = table_points[i][1]
            y2 = table_points[i + 1][1]
            result = (((y1 - y2) / (x1 - x2)) * point) + ((y2 * x1) - (y1 * x2)) / (x1 - x2)
            print(bcolors.BOLD, "\nThe approximation (interpolation) of the point ", point, " is: ",bcolors.ENDC, round(result, 4))
            return
    x1 = table_points[- 2][0]
    x2 = table_points[- 1][0]
    y1 = table_points[- 2][1]
    y2 = table_points[- 1][1]
    m = (y1 - y2) / (x1 - x2)
    b = y1 - m * x1
    result = m * point + b
    print(bcolors.BOLD, "\nThe approximation (extrapolation) of the point ", point, " is: ", bcolors.ENDC,
          round(result, 4))


if __name__ == '__main__':

    table_points = [(1, 3), (2, 4), (3, -1)]
    x = 1.5
    print(bcolors.OKBLUE, "----------------- Interpolation & Extrapolation Methods -----------------\n", bcolors.ENDC)
    print(bcolors.OKBLUE, "Table Points: ", bcolors.ENDC, table_points)
    print(bcolors.OKBLUE, "Finding an approximation to the point: ", bcolors.ENDC, x)
    linearInterpolation(table_points, x)
    print(bcolors.OKBLUE, "\n---------------------------------------------------------------------------\n", bcolors.ENDC)



