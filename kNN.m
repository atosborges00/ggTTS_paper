%% k Nearest Neighbours
% Code author: Atos Borges
% Date: 06/07/2021
% This code was developed and will publish by the authors at the paper:
% AUTOMATIC IDENTIFICATION OF SYNTHETICALLY GENARATED INTERLANGUAGE 
% TRANSFER PHENOMENA BETWEEN BRAZILIAN PORTUGUESE (L1) AND ENGLISH AS 
% FOREIGN LANGUAGE (L2).
%
% You SHOULD NOT use, copy, modify or redistribute any of this code before
% the final paper publication. We will let you know when this code will be
% available for public use under proper licensing.

%% Setup

% Cleaning and adding the subfolders to the MATLAB path
clear; close all; clc;
addpath(genpath('phonetic_data'))
addpath(genpath('kNN_functions'))
addpath(genpath('utils'))

% Setting default values for testing
train_percent = 70;
max_test_rounds = 100;
number_neighbours = 7;

%% Phenomenon HA

fprintf('\n\n HA PROCESS:\n')

% Loading data files
HA_X = load('HA_data.txt');
HA_Y = load('HA_targets.txt');

run_kNN(HA_X, HA_Y, number_neighbours, train_percent, max_test_rounds);

%% Phenomenon HAS-HPS

fprintf('\n\n HAS-PHS PROCESS:\n')

% Loading data files
HAS_HPS_X = load('HAS_HPS_data.txt');
HAS_HPS_Y = load('HAS_HPS_targets.txt');

run_kNN(HAS_HPS_X, HAS_HPS_Y, number_neighbours, train_percent, max_test_rounds);

%% Phenomenon KPVI

fprintf('\n\n KPVI PROCESS:\n')

% Loading data files
KPVI_X = load('KPVI_data.txt');
KPVI_Y = load('KPVI_targets.txt');

run_kNN(KPVI_X, KPVI_Y, number_neighbours, train_percent, max_test_rounds);

%% Phenomenon SSO

fprintf('\n\n SSO PROCESS:\n')

% Loading data files
SSO_X = load('SSO_data.txt');
SSO_Y = load('SSO_targets.txt');

run_kNN(SSO_X, SSO_Y, number_neighbours, train_percent, max_test_rounds);
