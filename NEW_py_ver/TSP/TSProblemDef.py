
import torch
import numpy as np
import math


def get_random_problems(batch_size, problem_size):
    problems = torch.rand(size=(batch_size, problem_size, 2))
    # problems.shape: (batch, problem, 2)
    return problems

# def get_random_problems(batch_size, problem_size): #単位円から作問
#     problems = torch.zeros(batch_size, problem_size, 2)
#     i, j = 0, 0
#     # for i in range(batch_size):
#     #   for j in range(problem_size):
#     while i < batch_size:
#       point = torch.rand(2)
#       # print(point)
#       # print(torch.sqrt(torch.sum(((point - 0.5)**2))).item())
#       if torch.sqrt(torch.sum(((point - 0.5)**2))).item() < 0.5:
#         problems[i, j] = point
#         j+=1
#         if j == problem_size:
#           i+=1
#           j = 0
#     # print(i, j)
#     return problems


def get_rotation_matrix(rad):
    rot = torch.tensor([[np.cos(rad), -np.sin(rad)],
                        [np.sin(rad), np.cos(rad)]])
    #print(type(rot))
    return rot


def rotate(xy, rad):
    xy = xy - 0.5
    batch = xy.shape[0]
    problem = xy.shape[1]
    xy = torch.reshape(xy, (-1, 2))
    for i in range(xy.shape[0]):
        xy[i] = torch.mv(get_rotation_matrix(rad).double(), (xy[i]).double())
    xy = xy + 0.5
    return torch.reshape(xy, (batch, problem, 2))


def augment_xy_data_by_4_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat(((1 - x), y), dim=2)
    dat3 = torch.cat((x, (1 - y)), dim=2)
    dat4 = torch.cat(((1 - x), (1 - y)), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems

def augment_xy_data_by_8_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat(((1 - x), y), dim=2)
    dat3 = torch.cat((x, (1 - y)), dim=2)
    dat4 = torch.cat(((1 - x), (1 - y)), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat(((1 - y), x), dim=2)
    dat7 = torch.cat((y, (1 - x)), dim=2)
    dat8 = torch.cat(((1 - y), (1 - x)), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems

#Translation Code

# def augment_xy_data_by_8_fold(problems):
#     # problems.shape: (batch, problem, 2)

#     x = problems[:, :, [0]]
#     y = problems[:, :, [1]]
#     # x,y shape: (batch, problem, 1)

    # dat1 = problems
    # dat2 = torch.cat((x-0.1, y-0.1), dim=2)
    # dat3 = torch.cat((x+0.1, y-0.1), dim=2)
    # dat4 = torch.cat((x-0.1, y+0.1), dim=2)
    # dat5 = torch.cat((x+0.2, y+0.2), dim=2)
    # dat6 = torch.cat((x-0.2, y-0.2), dim=2)
    # dat7 = torch.cat((x+0.2, y-0.2), dim=2)
    # dat8 = torch.cat((x-0.2, y+0.2), dim=2)

#     aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
#     # shape: (8*batch, problem, 2)

#     return aug_problems

# def augment_xy_data_by_10_fold(problems):
#     # problems.shape: (batch, problem, 2)

#     x = problems[:, :, [0]]
#     y = problems[:, :, [1]]
#     # x,y shape: (batch, problem, 1)

#     dat1 = torch.cat((x, y), dim=2)
#     dat2 = torch.cat(((1 - x), y), dim=2)
#     dat3 = torch.cat((x, (1 - y)), dim=2)
#     dat4 = torch.cat(((1 - x), (1 - y)), dim=2)
#     dat5 = torch.cat((y, x), dim=2)
#     dat6 = torch.cat(((1 - y), x), dim=2)
#     dat7 = torch.cat((y, (1 - x)), dim=2)
#     dat8 = torch.cat(((1 - y), (1 - x)), dim=2)
#     dat9 = torch.cat((x+0.1, y+0.1), dim=2)
#     dat10 = torch.cat((x-0.1, y-0.1), dim=2)

#     aug_problems = torch.cat(
#         (dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8, dat9, dat10), dim=0)
#     # shape: (10*batch, problem, 2)

#     return aug_problems


# def augment_xy_data_by_12_fold(problems):
#     # problems.shape: (batch, problem, 2)

#     x = problems[:, :, [0]]
#     y = problems[:, :, [1]]
#     # x,y shape: (batch, problem, 1)

#     dat1 = torch.cat((x, y), dim=2)
#     dat2 = torch.cat(((1 - x), y), dim=2)
#     dat3 = torch.cat((x, (1 - y)), dim=2)
#     dat4 = torch.cat(((1 - x), (1 - y)), dim=2)
#     dat5 = torch.cat((y, x), dim=2)
#     dat6 = torch.cat(((1 - y), x), dim=2)
#     dat7 = torch.cat((y, (1 - x)), dim=2)
#     dat8 = torch.cat(((1 - y), (1 - x)), dim=2)
#     dat9 = torch.cat((x+0.1, y+0.1), dim=2)
#     dat10 = torch.cat((x-0.1, y-0.1), dim=2)
#     dat11 = torch.cat((x+0.1, y-0.1), dim=2)
#     dat12 = torch.cat((x-0.1, y+0.1), dim=2)
#     aug_problems = torch.cat(
#         (dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8, dat9, dat10, dat11, dat12), dim=0)
#     # shape: (12*batch, problem, 2)

#     return aug_problems


# def augment_xy_data_by_16_fold(problems):
#     # problems.shape: (batch, problem, 2)

#     x = problems[:, :, [0]]
#     y = problems[:, :, [1]]
#     # x,y shape: (batch, problem, 1)

#     dat1 = torch.cat((x, y), dim=2)
#     dat2 = torch.cat(((1 - x), y), dim=2)
#     dat3 = torch.cat((x, (1 - y)), dim=2)
#     dat4 = torch.cat(((1 - x), (1 - y)), dim=2)
#     dat5 = torch.cat((y, x), dim=2)
#     dat6 = torch.cat(((1 - y), x), dim=2)
#     dat7 = torch.cat((y, (1 - x)), dim=2)
#     dat8 = torch.cat(((1 - y), (1 - x)), dim=2)
#     dat9 = torch.cat((x+0.1, y+0.1), dim=2)
#     dat10 = torch.cat((x-0.1, y-0.1), dim=2)
#     dat11 = torch.cat((x+0.1, y-0.1), dim=2)
#     dat12 = torch.cat((x-0.1, y+0.1), dim=2)
#     dat13 = torch.cat((x+0.2, y+0.2), dim=2)
#     dat14 = torch.cat((x-0.2, y-0.2), dim=2)
#     dat15 = torch.cat((x+0.2, y-0.2), dim=2)
#     dat16 = torch.cat((x-0.2, y+0.2), dim=2)

#     aug_problems = torch.cat(
#         (dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8, dat9, dat10, dat11, dat12, dat13, dat14, dat15, dat16), dim=0)
#     # shape: (16*batch, problem, 2)

#     return aug_problems

#Rotation Code

# def augment_xy_data_by_8_fold(problems):
#     # problems.shape: (batch, problem, 2)

#     dat1 = problems
#     dat2 = rotate(problems, -math.pi/12)
#     dat3 = rotate(problems, math.pi/6)
#     dat4 = rotate(problems, -math.pi/6)
#     dat5 = rotate(problems, math.pi/4)
#     dat6 = rotate(problems, -math.pi/4)
#     dat7 = rotate(problems, math.pi/3)
#     dat8 = rotate(problems, -math.pi/3)

#     aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
#     # shape: (8*batch, problem, 2)

#     return aug_problems

# def augment_xy_data_by_8_fold(problems):
#     # problems.shape: (batch, problem, 2)

#     dat1 = problems
#     dat2 = rotate(problems, math.pi*13/12)
#     dat3 = rotate(problems, math.pi/6)
#     dat4 = rotate(problems, math.pi*7/6)
#     dat5 = rotate(problems, math.pi/4)
#     dat6 = rotate(problems, math.pi*5/4)
#     dat7 = rotate(problems, math.pi/3)
#     dat8 = rotate(problems, math.pi*4/3)

#     aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
#     # shape: (8*batch, problem, 2)

#     return aug_problems

# def augment_xy_data_by_8_fold(problems):
#     # problems.shape: (batch, problem, 2)

#     dat1 = problems
#     dat2 = rotate(problems, math.pi/2)
#     dat3 = rotate(problems, -math.pi/2)
#     dat4 = rotate(problems, math.pi*4)
#     dat5 = rotate(problems, -math.pi/4)
#     dat6 = rotate(problems, math.pi*3/4)
#     dat7 = rotate(problems, -math.pi*3/4)
#     dat8 = rotate(problems, math.pi)

#     aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
#     # shape: (8*batch, problem, 2)

#     return aug_problems

def augment_xy_data_by_10_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat(((1 - x), y), dim=2)
    dat3 = torch.cat((x, (1 - y)), dim=2)
    dat4 = torch.cat(((1 - x), (1 - y)), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat(((1 - y), x), dim=2)
    dat7 = torch.cat((y, (1 - x)), dim=2)
    dat8 = torch.cat(((1 - y), (1 - x)), dim=2)
    dat9 = rotate(problems, math.pi/12)
    dat10 = rotate(problems, -math.pi/12)
    aug_problems = torch.cat(
        (dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8, dat9, dat10), dim=0)
    # shape: (10*batch, problem, 2)

    return aug_problems


def augment_xy_data_by_12_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat(((1 - x), y), dim=2)
    dat3 = torch.cat((x, (1 - y)), dim=2)
    dat4 = torch.cat(((1 - x), (1 - y)), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat(((1 - y), x), dim=2)
    dat7 = torch.cat((y, (1 - x)), dim=2)
    dat8 = torch.cat(((1 - y), (1 - x)), dim=2)
    dat9 = rotate(problems, math.pi/12)
    dat10 = rotate(problems, -math.pi/12)
    dat11 = rotate(problems, math.pi/6)
    dat12 = rotate(problems, -math.pi/6)
    aug_problems = torch.cat(
        (dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8, dat9, dat10, dat11, dat12), dim=0)
    # shape: (12*batch, problem, 2)

    return aug_problems


def augment_xy_data_by_16_fold(problems):
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat(((1 - x), y), dim=2)
    dat3 = torch.cat((x, (1 - y)), dim=2)
    dat4 = torch.cat(((1 - x), (1 - y)), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat(((1 - y), x), dim=2)
    dat7 = torch.cat((y, (1 - x)), dim=2)
    dat8 = torch.cat(((1 - y), (1 - x)), dim=2)
    dat9 = rotate(problems, math.pi/12)
    dat10 = rotate(problems, -math.pi/12)
    dat11 = rotate(problems, math.pi/6)
    dat12 = rotate(problems, -math.pi/6)
    dat13 = rotate(problems, math.pi/4)
    dat14 = rotate(problems, -math.pi/4)
    dat15 = rotate(problems, math.pi/3)
    dat16 = rotate(problems, -math.pi/3)

    aug_problems = torch.cat(
        (dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8, dat9, dat10, dat11, dat12, dat13, dat14, dat15, dat16), dim=0)
    # shape: (16*batch, problem, 2)

    return aug_problems
