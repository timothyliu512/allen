import pandas as pd
import numpy as np

from colormath.color_objects import sRGBColor, LabColor, XYZColor
from colormath.color_conversions import convert_color

from skimage import io, color
import skimage

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.colors import LinearSegmentedColormap

from scipy.spatial import ConvexHull


'''
Normalize RGB from 0-255 to 0-1
'''
def normalize_rgb_colors(rgb_colors):
    """
    Normalize an array of RGB colors.

    Args:
        rgb_colors (numpy.ndarray): An array of RGB color values.

    Returns:
        numpy.ndarray: An array of normalized RGB color values.
    """
    normalized_colors = rgb_colors / 255.0
    return np.array(normalized_colors)

####### Returns numpy array of numpy arrays of RGB values (fix later)

'''
RGB Array to Colormath LAB
'''
def rgb_arr_to_colormath_LAB(clr):
    """
    Converts an array of RGB values to colormath LAB color objects.

    Args:
        clr (list): A list of RGB color values in the format [[R, G, B], [R, G, B], ...].

    Returns:
        list: A list of colormath LAB color objects.
    """
    
    lab_colors = [] # Store lab color objects
    
    for i in range(len(clr)):
        rgb_color = sRGBColor(int(clr[i][0]), int(clr[i][1]), int(clr[i][2])) # Normalized RGB
        lab_color = convert_color(rgb_color, LabColor, target_illuminant='d50', observer='2') # Convert RGB to LAB
        lab_colors.append(lab_color)
    
    return np.array(lab_colors)


'''
RGB Array to Colormath RGB
'''
def rgb_arr_to_colormath_RGB(clr):
    """
    Converts an array of RGB values to colormath RGB and LAB color objects.

    Args:
        clr (list): A list of RGB color values in the format [[R, G, B], [R, G, B], ...].

    Returns:
        tuple: A tuple containing two lists. The first list contains colormath RGB color objects,
               and the second list contains colormath LAB color objects.
    """
    rgb_colors = [] #Store RGB color objects
    for i in range(len(clr)): #convert to RGB (normalized)
        rgb_color = sRGBColor(int(clr[i][0]), int(clr[i][1]), int(clr[i][2]))
        rgb_colors.append(rgb_color)
    
    return rgb_colors #Returns colormath objects

'''
RGB Array to LAB
'''
def RGB_arr_to_LAB_arr(clr):
    """
    Converts an array of RGB values to skimage RGB and LAB color objects.

    Args:
        clr (list): A list of RGB color values in the format [[R, G, B], [R, G, B], ...].

    Returns:
        tuple: A tuple containing two lists. The first list contains skimage LAB color objects,
               and the second list contains skimage LAB color objects.
    """
    
    # Convert to skimage objs from RGB
    lab_colors = []  # Store LAB color objects
    for i in range(len(clr)):
        lab_color = skimage.color.rgb2lab(clr[i], observer="2", illuminant="D50")
        lab_colors.append(lab_color)
    
    return lab_colors

'''
Is Valid RGB or LAB Color
'''
def is_valid_lab_and_rgb(lab_points, colormath_obj=False, in_fake_lab=False):
    """
    Check if LAB points are valid and convert to RGB to check if those are valid too.
    
    Parameters:
    - lab_points: numpy array of shape (n, 3) containing LAB points, or a list of LabColor objects
    - colormath_obj: Boolean indicating if lab_points is a list of colormath LabColor objects
    - in_fake_lab: Boolean indicating if LAB values need to be normalized
    
    Returns:
    - Boolean indicating whether the LAB points are valid and result in valid RGB values upon conversion.
    """
    
    if colormath_obj:
        # If lab_points is already a list of LabColor objects, use it directly
        lab_colors = lab_points
        if in_fake_lab:
            # Normalize L channel
            for color in lab_colors:
                color.lab_l = color.lab_l + 50
    else:
        # Assume lab_points is a numpy array and convert to LabColor objects
        lab_points2 = np.copy(lab_points)  # Copy to avoid modifying original
        if in_fake_lab:
            # Normalize L channel
            lab_points2[:, 0] = lab_points2[:, 0] + 50  # Adjust if in "fake" LAB space
        lab_colors = [LabColor(lab_l=point[0], lab_a=point[1], lab_b=point[2]) for point in lab_points2]
    
    # Convert LAB numpy array to colormath LabColor objects and then to sRGBColor objects
    for lab_point in lab_colors:
        if lab_point.lab_l > 100 or lab_point.lab_l < 0 or np.abs(lab_point.lab_a) > 127 or np.abs(lab_point.lab_b) > 127:
            print(lab_point)
            return False
        
        rgb_color = convert_color(lab_point, sRGBColor, target_illuminant='d50', observer='2')

        # Extract RGB values to check their validity (sRGBColor values are between 0 and 1)
        r, g, b = rgb_color.get_value_tuple()

        # Check if the converted RGB values are within the valid range
        if not (0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1):
            print(rgb_color)
            return False

    return True

def LAB_arr_to_LAB_Colormath(lab_arr, fake_lab=False):
    """
    Convert an array of LAB values to an array of RGB values using colormath.

    Args:
        lab_arr (numpy.ndarray): An array of LAB color values.
        fake_lab (bool): Whether the LAB values are in the fake LAB color space (default: False).

    Returns:
        numpy.ndarray: An array of RGB color values.
    """
    # Copy to avoid modifying original
    lab_points = np.copy(lab_arr)
    
    if fake_lab:
        # Normalize L channel
        lab_points[:, 0] = (lab_points[:, 0] + 50)  # Convert back to real LAB values if needed
    
    lab_colormath = []
    
    for point in lab_points:
        lab_color = LabColor(lab_l=point[0], lab_a=point[1], lab_b=point[2])
        lab_colormath.append(lab_color)
    
    return lab_colormath



'''
Color Gradient Plot
'''
def plot_color_gradient(colors, plot_title=None):
    """
    Plot a color gradient using the given colors.

    Parameters:
    colors (list): A list of RGB tuples representing the colors. Must be normalized
    plot_title (str): The title of the plot.

    Returns:
    None
    """
    
    num_colors = len(colors)

    gradient = LinearSegmentedColormap.from_list('color_gradient', colors)

    plt.imshow(np.arange(num_colors).reshape(1, num_colors), cmap=gradient, aspect='auto')
    if plot_title == None:
        plt.title(str(num_colors)+' Colors')
    else:
        plt.title(plot_title)
    plt.axis('off')
    plt.show()
    
    
    
'''
Plot 3D Hull
'''

def plot_hull3D(lab_clrs, plot_title="Convex Hull", colormath_obj=False, save_filepath=None):
    """
    Plots Lab colors with a convex hull in 3D.

    Parameters:
    lab_clrs (numpy.ndarray or list): Array of Lab colors or a list of LabColor objects.
    plot_title (str): Title of the plot.
    colormath_obj (bool): Indicates if lab_clrs is a list of LabColor objects.
    save_filepath (str, optional): Path to save the plot image.

    Returns:
    None
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if colormath_obj:
        # Convert LabColor objects to a numpy array for plotting and hull calculation
        lab_array = np.array([[color.lab_l, color.lab_a, color.lab_b] for color in lab_clrs])
    else:
        lab_array = np.array(lab_clrs)

    # Calculate the convex hull
    hull = ConvexHull(lab_array)

    # Plot the original Lab colors
    for color in lab_array:
        rgb_color = convert_color(LabColor(lab_l=color[0], lab_a=color[1], lab_b=color[2]), sRGBColor).get_upscaled_value_tuple()
        rgb_color = [max(0, min(1, c/255)) for c in rgb_color]  # Normalize and clamp
        ax.scatter(color[1], color[2], color[0], c=[rgb_color], marker='o')  # Plot using Lab a, b, L as coordinates

    # Plot Convex Hull in 3D
    for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0])  # Close loop
        ax.plot(lab_array[simplex, 1], lab_array[simplex, 2], lab_array[simplex, 0], 'r-')
        
    ax.set_title(plot_title, weight='bold')
    ax.set_xlabel('A*')
    ax.set_ylabel('B*')
    ax.set_zlabel('L*')
    fig.set_size_inches(6, 6)

    # Set x, y, and z axes limits
    ax.set_xlim(-128, 128)
    ax.set_ylim(-128, 128)
    ax.set_zlim(0, 100)  # L* ranges from 0 to 100

    if save_filepath:
        plt.savefig(save_filepath, dpi=300)
    plt.show()

# Example usage:
# Assuming lab_clrs is a numpy array of Lab values
# plot_hull3D(lab_clrs, colormath_obj=False)

    
    
'''
Plot Colormath LAB in 3D
'''

def plot_LAB_3D(lab_clrs, title="Lab Colors in 3D", clamp_RGB=False, colormath_obj=False, save_filepath=None):
    """
    Plots LAB color objects in a 3D scatter plot.

    Parameters:
    lab_colors (list): List of LAB color objects.
    title (str): Title of the plot.
    clamp_rgb (bool): Whether to clamp the RGB values to the valid range (default: False).
    save_filepath (str): Filepath to save the plot (default: None).

    Returns:
    None
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    print(f"RGB Color Coordinates:")
    
    if not colormath_obj:
        lab_colors = LAB_arr_to_LAB_Colormath(lab_clrs)
        print(lab_colors)
        print(type(lab_colors))
    else:
        lab_colors = lab_clrs
    
    #Plot LAB colors
    for color in lab_colors:
        
        #xyz_color = convert_color(color, XYZColor, observer='2', illuminant ='d50') #Convert from cielab to XYZ
        rgb_color = convert_color(color, sRGBColor, observer='2', illuminant ='d50').get_upscaled_value_tuple() #Convert from RGB to LAB
        
        #Normalize rgb_color to 0-1
        #rgb_color_normalized = [c / 255 for c in rgb_color]
        
        # Clamp RGB values to valid range if clamp_rgb is True
        if clamp_RGB:
            rgb_color = [max(0, min(1, c)) for c in rgb_color]
        
        rgb_color = [color / 255 for color in rgb_color]
        
        
        #Plot lab colors, and color them with their normalized RGB values
        ax.scatter(color.lab_a, color.lab_b, color.lab_l, c=[rgb_color]) 

    ax.set_title(title, weight='bold')
    ax.set_xlabel('A* Label')
    ax.set_ylabel('B* Label')
    ax.set_zlabel('L* Label')
    ax.figure.set_size_inches(6, 6)

    # Set x, y, and z axes
    ax.set_xlim(-128, 128)
    ax.set_ylim(-128, 128)
    ax.set_zlim(-128, 128)

    if save_filepath:
        plt.savefig(save_filepath, dpi=300)
    plt.show() 
    
'''
Plot 2D LAB
'''

def plot_LAB_2D(lab_input, title="LAB Colors in 2D", axes=('l', 'a'), clamp_RGB=True, show_RGB=True, show_LAB=False, colormath_obj = False, save_filepath=None):
    
    """
    Plot 2D LAB color objects in a specified 2D plane.

    Args:
        lab_colors (list): List of LabColor objects.
        title (str): Title of the plot.
        axes (tuple): Tuple specifying the two axes to be displayed, case-insensitive ('l', 'a', 'b').
        clamp_rgb (bool): Whether to clamp the RGB values to the valid range (default: True).

    Returns:
        None
    """
    # Normalize axes input to lowercase
    axes = tuple(axis.lower() for axis in axes)

    # Validate axes
    valid_axes = {'l', 'a', 'b'}
    if not set(axes).issubset(valid_axes) or len(axes) != 2:
        raise ValueError("Invalid axes. Specify any two from 'L', 'a', 'b', case-insensitive.")

    # Start plotting
    fig, ax = plt.subplots()
    
    if show_RGB:
        print(f"RGB Color Coordinates:")

    if not colormath_obj:
        lab_colormath = LAB_arr_to_LAB_Colormath(lab_input)
    else:
        lab_colormath = lab_input
        
    for color in lab_colormath:
        # Convert LAB to RGB for plotting colors
        #xyz_color = convert_color(color, XYZColor, target_illuminant='d50', observer='2')  # Convert from CIELAB to XYZ
        rgb_color = convert_color(color, sRGBColor, target_illuminant='d50', observer='2')  # Convert from XYZ to sRGB
        
        if clamp_RGB:
            rgb_values = (rgb_color.clamped_rgb_r, rgb_color.clamped_rgb_g, rgb_color.clamped_rgb_b)
        else:
            rgb_values = (rgb_color.rgb_r, rgb_color.rgb_g, rgb_color.rgb_b)

        # Determine plot coordinates based on axes
        coord_map = {'l': color.lab_l, 'a': color.lab_a, 'b': color.lab_b}
        x, y = coord_map[axes[0]], coord_map[axes[1]]

        if show_RGB:
            print(rgb_color)
        ax.scatter(x, y, color=[rgb_values])
        if show_LAB:
            print(f" {color.lab_l},  {color.lab_a},  {color.lab_b}")

    # Set plot titles and labels
    ax.set_title(title)
    ax.set_xlabel(f"{axes[0].upper()}*")
    ax.set_ylabel(f"{axes[1].upper()}*")

    # Set limits based on axes chosen
    if 'l' in axes:
        ax.set_ylim(0, 100) if axes[1] == 'l' else ax.set_xlim(0, 100)
    else:
        ax.set_ylim(-128, 128)
        ax.set_xlim(-128, 128)

    if show_LAB:
        print(f"LAB Color Coordinates:")
    if save_filepath:
        plt.savefig(save_filepath, dpi=300)
    plt.show()


'''
LAB to RGB
'''

def lab_to_rgb(lab_color, colormath=False):
    """
    Convert LAB color to RGB color.

    Parameters:
    lab_color (tuple): Tuple containing LAB color values (L, a, b).

    Returns:
    tuple: Tuple containing RGB color values (R, G, B).
    """

    # Create LabColor object
    if not colormath:
        lab_colormath = LAB_arr_to_LAB_Colormath(lab_color)
    else: 
        lab_colormath = lab_color

    rgb_colors = []
    for color in lab_colormath:
        rgb_clr = convert_color(color, sRGBColor, observer='2', target_illuminant='d50')
        rgb_colors.append(rgb_clr.get_value_tuple())

    # Return RGB color
    return np.array(rgb_colors)

def save_colors(colors, filepath, upscale_to_255=False):
    # Convert colors to a numpy array
    if upscale_to_255:
        colors_array = [(r * 255, g * 255, b * 255) for r, g, b in colors]
    
    colors_array = np.array(colors)
    
    # Save the array to a .txt file
    np.savetxt(filepath, colors_array, fmt='%s')
    
def plot_dual_hull3D(lab_clrs1, lab_clrs2, plot_title="Dual Convex Hulls", colormath_obj=False, hull_colors=('r', 'g'), save_filepath=None):
    """
    Plots two sets of Lab colors with their convex hulls in 3D in different colors.

    Parameters:
    lab_clrs1, lab_clrs2 (numpy.ndarray or list): Arrays of Lab colors or lists of LabColor objects for two different sets.
    plot_title (str): Title of the plot.
    colormath_obj (bool): Indicates if lab_clrs is a list of LabColor objects.
    hull_colors (tuple): A tuple containing the colors for the convex hull lines of the first and second set.
    save_filepath (str, optional): Path to save the plot image.

    Returns:
    None
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def plot_hull(lab_clrs, hull_color):
        if colormath_obj:
            lab_array = np.array([[color.lab_l, color.lab_a, color.lab_b] for color in lab_clrs])
        else:
            lab_array = np.array(lab_clrs)

        hull = ConvexHull(lab_array)

        for color in lab_array:
            rgb_color = convert_color(LabColor(lab_l=color[0], lab_a=color[1], lab_b=color[2]), sRGBColor, observer='2', target_illuminant='d50').get_upscaled_value_tuple()
            rgb_color = [max(0, min(1, c/255)) for c in rgb_color]  # Normalize and clamp
            ax.scatter(color[1], color[2], color[0], c=[rgb_color], marker='o')

        for simplex in hull.simplices:
            simplex = np.append(simplex, simplex[0])  # Close loop
            ax.plot(lab_array[simplex, 1], lab_array[simplex, 2], lab_array[simplex, 0], color=hull_color, linestyle='-')


    # Plot the first and second set of Lab colors with their convex hulls
    plot_hull(lab_clrs1, hull_colors[0])
    plot_hull(lab_clrs2, hull_colors[1])

    ax.set_title(plot_title, weight='bold')
    ax.set_xlabel('A*')
    ax.set_ylabel('B*')
    ax.set_zlabel('L*')
    fig.set_size_inches(6, 6)

    ax.set_xlim(-128, 128)
    ax.set_ylim(-128, 128)
    ax.set_zlim(0, 100)  # L* ranges from 0 to 100

    if save_filepath:
        plt.savefig(save_filepath, dpi=300)
    plt.show()
    


def plot_dual_hull2D(lab_clrs1, lab_clrs2, axes=('a', 'b'), plot_title="Dual Convex Hulls", colormath_obj=False, hull_colors=('r', 'g'), save_filepath=None):
    """
    Plots two sets of Lab colors with their convex hulls in 2D in different colors, allowing selection of axes.

    Parameters:
    lab_clrs1, lab_clrs2 (numpy.ndarray or list): Arrays of Lab colors or lists of LabColor objects for two different sets.
    axes (tuple): A tuple containing two characters representing the axes to plot ('l', 'a', or 'b').
    plot_title (str): Title of the plot.
    colormath_obj (bool): Indicates if lab_clrs is a list of LabColor objects.
    hull_colors (tuple): A tuple containing the colors for the convex hull lines of the first and second set.
    save_filepath (str, optional): Path to save the plot image.

    Returns:
    None
    """

    fig, ax = plt.subplots()

    # Axis indices based on 'l', 'a', 'b' selection
    axis_map = {'l': 0, 'a': 1, 'b': 2}
    axis_indices = [axis_map[axes[0]], axis_map[axes[1]]]

    def plot_hull(lab_clrs, hull_color):
        if colormath_obj:
            lab_array = np.array([[color.lab_l, color.lab_a, color.lab_b] for color in lab_clrs])
        else:
            lab_array = np.array(lab_clrs)

        hull = ConvexHull(lab_array[:, axis_indices])

        for color in lab_array:
            rgb_color = convert_color(LabColor(lab_l=color[0], lab_a=color[1], lab_b=color[2]), sRGBColor, observer='2', target_illuminant='d50').get_upscaled_value_tuple()
            rgb_color = [max(0, min(1, c/255)) for c in rgb_color]  # Normalize and clamp
            ax.scatter(color[axis_indices[0]], color[axis_indices[1]], c=[rgb_color], marker='o')

        for simplex in hull.simplices:
            simplex = np.append(simplex, simplex[0])  # Close loop
            ax.plot(lab_array[simplex, axis_indices[0]], lab_array[simplex, axis_indices[1]], color=hull_color, linestyle='-')

    # Plot the first and second set of Lab colors with their convex hulls
    plot_hull(lab_clrs1, hull_colors[0])
    plot_hull(lab_clrs2, hull_colors[1])

    ax.set_title(plot_title, weight='bold')
    ax.set_xlabel(f'{axes[0].upper()}*')
    ax.set_ylabel(f'{axes[1].upper()}*')
    fig.set_size_inches(6, 6)

    if axes[0] in ['a', 'b'] and axes[1] in ['a', 'b']:
        ax.set_xlim(-128, 128)
        ax.set_ylim(-128, 128)
    else:
        ax.set_xlim(0, 100) if 'l' in axes[0] else ax.set_xlim(-128, 128)
        ax.set_ylim(-128, 128) if 'l' not in axes[1] else ax.set_ylim(0, 100)

    if save_filepath:
        plt.savefig(save_filepath, dpi=300)
    plt.show()

