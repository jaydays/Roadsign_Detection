import io, sys, types
import os
import shutil

curDir = os.getcwd()
# unorganized_db_path = input("Enter unorganized DB path: ")
# print(unorganized_db_path)
# exit()

new_db_path = os.path.join(curDir, "DataBase")
original_db_path = os.path.join(curDir, "signDatabasePublicFramesOnly")

dataFolder = "Data"
newAnnotationFile = "annotations.csv"
oldAnnotationFile = "allAnnotations.csv"  # "testAnnotations.csv"
semiColon = ";"

annotationPath = os.path.join(original_db_path, oldAnnotationFile)
if not os.path.isfile(annotationPath):
    print("The given annotation file does not exist.")
    exit()

csv = open(os.path.abspath(annotationPath), 'r')

header = csv.readline()
fields = header.split(";")
selectedElements = ["number of signs", fields[0], fields[1], fields[2], fields[3], fields[4], fields[5]]
header = semiColon.join(selectedElements)

csv = csv.readlines()
csv.sort()

allAnnotations = []

basePath = new_db_path
savePath = os.path.join(basePath, dataFolder)

if not os.path.isdir(basePath):
    os.mkdir(basePath)
if not os.path.isdir(savePath):
    os.mkdir(savePath)

counter = 0
previousFile = ''
numSigns = 0
maxNumSigns = 0
for line in csv:
    fields = line.split(";")

    if previousFile != fields[0]:
        newFile = fields[0][fields[0].rfind("/") + 1:]
        selectedElements = [newFile, fields[1], fields[2], fields[3], fields[4], fields[5]]
        newLine = "\n" + semiColon.join(selectedElements)
        allAnnotations.append(newLine)
        numSigns = 1
        shutil.copy(os.path.join(original_db_path, fields[0]), savePath)

    else:
        selectedElements = [fields[1], fields[2], fields[3], fields[4], fields[5]]
        newLine = ";" + semiColon.join(selectedElements)
        allAnnotations.append(newLine)
        numSigns += 1

    if numSigns > maxNumSigns:
        maxNumSigns = numSigns
    previousFile = fields[0]
    counter += 1
    continue

out = open(os.path.join(basePath, newAnnotationFile), 'w')

out.write(header)
out.writelines(allAnnotations)
out.close()

print(maxNumSigns)
print("Done. Processed %d annotations." % (counter + 1))
