"""
DSC 20 Mid-Quarter Project
Name: Amogh Patankar
PID:  A16580842
"""

# Part 1: RGB Image #
class RGBImage:
    """
    This class defines an RGB image, including initialization, 
    sizing, returns an individual and all the pixels in the image,
    as well as image copies and changing individual pixel colors.
    """

    def __init__(self, pixels):
        """
        This is a constructor for the RGBImage class, and it initilizes
        the instance variable pixels to the parameter passed value.
        """
        
        self.pixels = pixels  # initialze the pixels list here

    def size(self):
        """
        This function returns the size of the image, in a tuple format.
        The tuple is constructed in the format (rows, columns). 
        """

        return (len(self.pixels[0]), len(self.pixels[0][0]))

    def get_pixels(self):
        """
        This function returns a deep copy of the pixels matrix as a 3D list.
        This matrix is the exact same as pixels, but a deepcopy. 
        """

        return [[[c for c in r] for r in ch] for ch in self.pixels]

    def copy(self):
        """
        This function returns a copy of a different RGBImage instance. This 
        instance is also initialized with a deep copy of the pixels matrix.
        """
                
        return RGBImage(self.get_pixels())

    def get_pixel(self, row, col):
        """
        This function returns the color of a patricular pixel at a given
        position-- at (row, col). The return type is a 3-element tuple, 
        as all colors are given in a red-green-blue (RGB) intensity format. 
        """

        assert isinstance(row, int)
        assert isinstance(col, int)
        assert row >= 0
        assert col >= 0

        max_colors = 3
        return tuple(self.pixels[i][row][col] for i in range(max_colors))

        #for i in range(max_colors):
            #sol.append(self.pixels[i][row][col])
        #return tuple(sol)

    def set_pixel(self, row, col, new_color):
        """
        This function changes/updates the color of a particular pixel given
        the row, column, and new_color value. If any of the color intensities
        are -1 (red, green, or blue), the channel value does not change
        """
        
        assert isinstance(row, int)
        assert isinstance(col, int)
        assert row >= 0
        assert col >= 0
        assert row <= self.size()[0]
        assert col <= self.size()[1]

        for i in range(len(new_color)):
            if new_color[i] != -1:
                self.pixels[i][row][col] = new_color[i]


# Part 2: Image Processing Methods #
class ImageProcessing:
    """
    This class implements a multitude of image process methods, including
    negate, grayscale, removing red/green/blue (single or combo), cropping,
    chrome backgrounds, and rotation. 
    """

    @staticmethod
    def negate(image):
        """
        This function negates the color in a given image. The colors are
        negated by subtracting the color value of individual pixels from
        255. 
        """
        
        assert isinstance(image, RGBImage)

        temp_matrix = image.get_pixels()
        max_val = 255
        return RGBImage([[[max_val - temp_matrix[chan][row][col] for col in\
         range(len(temp_matrix[chan][row]))] for row in\
          range(len(temp_matrix[chan]))] for chan in\
           range(len(temp_matrix))])

    @staticmethod
    def grayscale(image):
        """
        This function converts all the pixels to grayscale, and does so by
        taking the average of the RGB values of each individual pixel. 
        """

        assert isinstance(image, RGBImage)

        return RGBImage([[[sum(image.get_pixel(r, c)) // 3 for c in\
         range(image.size()[1])] for r in range(image.size()[0])] for\
          a in range(3)])

    @staticmethod
    def clear_channel(image, channel):
        """
        This function accepts an RGBImage object and a channel as parameters
        and clears the particular channel and returns the RGBImage
        """

        assert isinstance(image, RGBImage)
        assert isinstance(channel, int)
        
        temp_matrix = image.get_pixels()
        temp_matrix[channel] = [[0 for col in\
         range(len(temp_matrix[channel][row]))] for row in\
          range(len(temp_matrix[channel]))]

        return RGBImage(temp_matrix)  

    @staticmethod
    def crop(image, tl_row, tl_col, target_size):
        """
        This function crops an image given a starting row and column.
        If the newly cropped image exceeds the max height and width, 
        the function restricts the values to the max height and width.
        """

        temp_matrix = image.get_pixels()
        n_rows = target_size[0]
        n_cols = target_size[1]
        if tl_row + n_rows > len(temp_matrix[0]) and\
         tl_col + n_cols > len(temp_matrix[0][0]):
            return RGBImage([[[temp_matrix[chan][row][col] for col in\
             range(tl_col, len(temp_matrix[chan][row]))] for row in\
              range(tl_row, len(temp_matrix[chan]))] for chan in\
               range(len(temp_matrix))])

        elif tl_row + n_rows > len(temp_matrix[0]):
            return RGBImage([[[temp_matrix[chan][row][col] for col in\
             range(tl_col, tl_col + n_cols)] for row in\
              range(tl_row, len(temp_matrix[chan]))] for chan in\
               range(len(temp_matrix))])

        elif tl_col + n_cols > len(temp_matrix[0][0]): 
            return RGBImage([[[temp_matrix[chan][row][col] for col in\
             range(tl_col, len(temp_matrix[chan][row]))] for row in\
              range(tl_row, tl_row + n_rows)] for chan in\
               range(len(temp_matrix))])
        else:
            return RGBImage([[[temp_matrix[chan][row][col] for col in\
             range(tl_col, tl_col + n_cols)] for row in\
              range(tl_row, tl_row + n_rows)] for chan in\
               range(len(temp_matrix))])       

    @staticmethod
    def chroma_key(chroma_image, background_image, color):
        """
        This function checks if the chroma_image's color matches
        the color passed as a parameter at a particular pixel. 
        If it matches, then the background image's pixel's color is set
        to the color parameter. The resultant RGBImage blends chroma and 
        background images. 
        """
        
        assert isinstance(chroma_image, RGBImage)
        assert isinstance(background_image, RGBImage)
        assert chroma_image.size() == background_image.size()

        temp_matrix = chroma_image.get_pixels()
        bkgd = background_image.get_pixels()
        for row in range(len(temp_matrix[0])):
            for col in range(len(temp_matrix[0][row])):
                temp_color = list()
                for chan in range(3):
                    temp_color.append(temp_matrix[chan][row][col])
                if tuple(temp_color) == color:
                    for chan in range(3):
                        temp_matrix[chan][row][col] = bkgd[chan][row][col]

        return RGBImage(temp_matrix)

    # rotate_180 IS FOR EXTRA CREDIT (points undetermined)
    @staticmethod
    def rotate_180(image):
        """
        The function rotates an image 180˚ clockwise, and returns an 
        RGBImage. 
        """
        
        temp = image.get_pixels()
        num_chan = 3

        sol = [[[temp[chan][row][col] for col in range(len(temp[chan]\
            [row])-1, -1, -1)] for row in range(len(temp[chan])-1, -1, -1)]\
             for chan in range(num_chan)]
             
        return RGBImage(sol)
        
        
# Part 3: Image KNN Classifier #
class ImageKNNClassifier:
    """
    This class predicts the label for an image by finding popular
    labels in a collection of training data.
    """

    def __init__(self, n_neighbors):
        """
        This function initializes the n_neighbors value, which indicates
        the size of the nearest neighborhood, which is the number of
        neighbors required to make a prediction
        """
        
        self.n_neighbors = n_neighbors
        self.data = []

    def fit(self, data):
        """
        This function stores the training data in the instance of the 
        classifier. Data is a list of tuples (of RGBImages and label)
        """
        
        assert len(data) > self.n_neighbors
        assert len(self.data) == 0
        
        self.data = data

    @staticmethod
    def distance(image1, image2):
        """
        This function calculates the Euclidean distance between two 
        RGBImages (image1, image2). 
        """
        
        assert isinstance(image1, RGBImage)
        assert isinstance(image2, RGBImage)
        assert image1.size() == image2.size()

        return sum([sum([sum([(image1.pixels[chan][row][col]\
         - image2.pixels[chan][row][col]) ** 2 for col in\
          range(image1.size()[1])]) for row in range(image1.size()[0])])\
           for chan in range(3)]) ** (1/2)

    @staticmethod
    def vote(candidates):
        """
        This function finds the most popular label from the list
        of candidates' i.e. nearest neighbor labels. Returns any
        of the labels if there is a tie. 
        """
        
        mc = {}
        for i in candidates:
            tl = i[2]
            if tl in mc:
                mc[tl] += 1
            else:
                mc[tl] = 1

        return mc

    def predict(self, image):
        """
        The function predicts the label of a given image using KNN
        Classification, using the vote() function to predict using the 
        nearest neighbors.
        """
        
        assert len(self.data) > 0

        d = [(ImageKNNClassifier.distance(image, i[0]), i[0],\
         i[1]) for i in self.data]
        d = sorted(d)
        kn = d[:self.n_neighbors]
        return max(ImageKNNClassifier.vote(kn))