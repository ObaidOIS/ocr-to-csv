from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from pytesseract import Output
from tabulate import tabulate
import pandas as pd
import numpy as np
import pytesseract
import cv2


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# img = cv2.imread("333.jpg")
# print(pytesseract.image_to_boxes(img))


def plt_imshow(title, image):
    # convert the image frame BGR to RGB color space and display it
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(title)
    plt.grid(False)
    plt.show()


def convert(img):
    # since we are using Jupyter Notebooks we can replace our argument
    # parsing code with *hard coded* arguments and values # michael_jordan_stats.png # atevH.jpg
    # image = cv2.imread(image)
    # plt.imshow(image)
    # pil_image = PIL.image.convert('RGB')
    # open_cv_image = np.array(pil_image) 
    # # Convert RGB to BGR 
    # open_cv_image = open_cv_image[:, :, ::-1].copy()  
    # image = np.array(image)
    print('in convert')
    args = {
        "image": "333.jpg",
        "output": "results.csv",
        "min_conf": 0,
        "dist_thresh": 25.0,
        "min_size": 2,
    }

    input_img = img
    image = img
    # set a seed for our random number generator
    np.random.seed(42)
    table = image
    # set the PSM mode to detect sparse text, and then localize text in
    # the table
    options = "--psm 6"
    results = pytesseract.image_to_data(
        cv2.cvtColor(table, cv2.COLOR_BGR2RGB), config=options, output_type=Output.DICT
    )
    # print(results)
    # initialize a list to store the (x, y)-coordinates of the detected
    # text along with the OCR'd text itself
    coords = []
    ocrText = []
    # loop over each of the individual text localizations
    for i in range(0, len(results["text"])):
        # extract the bounding box coordinates of the text region from
        # the current result
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]
        # extract the OCR text itself along with the confidence of the
        # text localization
        text = results["text"][i]
        conf = int(float(results["conf"][i]))
        # filter out weak confidence text localizations
        if conf > args["min_conf"]:
            # update our text bounding box coordinates and OCR'd text,
            # respectively
            coords.append((x, y, w, h))
            ocrText.append(text)

    # extract all x-coordinates from the text bounding boxes, setting the
    # y-coordinate value to zero
    xCoords = [(c[0], 0) for c in coords]
    print()
    # apply hierarchical agglomerative clustering to the coordinates
    clustering = AgglomerativeClustering(
        n_clusters=None,
        affinity="manhattan",
        linkage="complete",
        distance_threshold=args["dist_thresh"],
    )
    clustering.fit(xCoords)


    # initialize our list of sorted clusters
    sortedClusters = []

    # loop over all clusters
    for l in np.unique(clustering.labels_):
        # extract the indexes for the coordinates belonging to the
        # current cluster
        idxs = np.where(clustering.labels_ == l)[0]

        # verify that the cluster is sufficiently large
        if len(idxs) > args["min_size"]:
            # compute the average x-coordinate value of the cluster and
            # update our clusters list with the current label and the
            # average x-coordinate
            avg = np.average([coords[i][0] for i in idxs])
            sortedClusters.append((l, avg))

    # sort the clusters by their average x-coordinate and initialize our
    # data frame
    sortedClusters.sort(key=lambda x: x[1])
    df = pd.DataFrame()

    # loop over the clusters again, this time in sorted order
    for (l, _) in sortedClusters:
        # extract the indexes for the coordinates belonging to the
        # current cluster
        idxs = np.where(clustering.labels_ == l)[0]

        # extract the y-coordinates from the elements in the current
        # cluster, then sort them from top-to-bottom
        yCoords = [coords[i][1] for i in idxs]
        sortedIdxs = idxs[np.argsort(yCoords)]

        # generate a random color for the cluster
        color = np.random.randint(0, 255, size=(3,), dtype="int")
        color = [int(c) for c in color]

        # loop over the sorted indexes
        for i in sortedIdxs:
            # extract the text bounding box coordinates and draw the
            # bounding box surrounding the current element
            (x, y, w, h) = coords[i]
            cv2.rectangle(table, (x, y), (x + w, y + h), color, 2)

        # extract the OCR'd text for the current column, then construct
        # a data frame for the data where the first entry in our column
        # serves as the header
        cols = [ocrText[i].strip() for i in sortedIdxs]
        currentDF = pd.DataFrame({cols[0]: cols[1:]})

        # concatenate *original* data frame with the *current* data
        # frame (we do this to handle columns that may have a varying
        # number of rows)
        df = pd.concat([df, currentDF], axis=1)


    # replace NaN values with an empty string and then show a nicely
    # formatted version of our multi-column OCR'd text
    df.fillna("", inplace=True)
    # print(tabulate(df, headers="keys", tablefmt="psql"))
    # print(df)
    # write our table to disk as a CSV file
    print("[INFO] saving CSV file to disk...")
    # df.to_csv(args["output"], index=False)
    # plt_imshow("Output", image)
    return df
    # show the output image after performing multi-column OCR

