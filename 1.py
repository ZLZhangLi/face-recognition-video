#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import caffe

if __name__ == "__main__":
    inputFolder = './input'
    inputMeshFormat ='*.* '
    outputFolder = './output'
    outputMeshFormat = '.ply'
    #mlxScriptFile = './mlx/'
    mlxScriptFolder = './mlx2/'
    outputMeshOptions = '-m vc fq wn'
    meshlabserverPath = 'D:/DATA1/VCG/MeshLab/meshlabserver.exe'
    #for % % X in ( % inputFolder % \ *.% inputMeshFormat % ) do (echo "%%X")
    files = os.listdir(inputFolder)
    for file in files:
        mlxScriptFiles = os.listdir(mlxScriptFolder)
        for mlxScriptFile in mlxScriptFiles:
            inputfolder = inputFolder + '/' + file
            outputfolder = outputFolder + '/' + file[:-4] + outputMeshFormat
            mlxScriptFilefolder = mlxScriptFolder + str(mlxScriptFiles)[2:-2]
            command_mean = meshlabserverPath + ' ' + '-i' + ' ' + inputfolder + ' ' + '-o' + ' ' + outputfolder + ' ' + outputMeshOptions + ' ' + '-s' + ' ' + mlxScriptFilefolder
            #print command_mean
            os.system(command_mean)