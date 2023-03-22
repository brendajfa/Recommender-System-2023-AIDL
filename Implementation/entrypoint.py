
import sys
from main_docker import Main



#     main = Main()
#     main.start()
#     exit()
projectname = "Recommender System of Music Intrumentals with several models"
message = f"\n\n   Welcome to our final project: {projectname}   \n" 
title = "_"*len(message)+message+"_"*len(message)
explication = ["\nThe objective of this project is to analyze the responses of different models of the aforementioned \ndataset.",
               " To compare the results, the 'Movie lens' dataset has been used as a reference.",
               "That is why \nright now we give you the option to run the code with the dataset you want:",
               "\n\t- MovieLens\n\t- Amazon"]


if sys.argv[1] == "Amazon":
    main = Main()
    main.start()
    exit()

elif sys.argv[1] == "MovieLens":
    main = Main("movie lens")
    main.start()
    exit()
else:
    print(title)
    print(explication[0]+explication[1]+explication[2]+explication[3])