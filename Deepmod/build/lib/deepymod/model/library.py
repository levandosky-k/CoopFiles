""" Contains the library classes that store the parameters u_t, theta"""
import numpy as np
import torch
from torch.autograd import grad
from itertools import combinations
from functools import reduce
from .deepmod import Library
from typing import Tuple
from ..utils.types import TensorList


# ==================== Library helper functions =======================
def library_poly(prediction: torch.Tensor, max_order: int) -> torch.Tensor:
    """Given a prediction u, returns u^n up to max_order, including ones as first column.

    Args:
        prediction (torch.Tensor): the data u for which to evaluate the library (n_samples x 1)
        max_order (int): the maximum polynomial order up to which compute the library

    Returns:
        torch.Tensor: Tensor with polynomials (n_samples, max_order + 1)
    """
    u = torch.ones_like(prediction) #tensor filled with 1s of the same size as prediction
    for order in np.arange(1, max_order + 1): #evenly spaced values in interval (counts from 1 to max_order + 1)
        u = torch.cat((u, u[:, order - 1 : order] * prediction), dim=1) 

    return u


def library_deriv(
    data: torch.Tensor, prediction: torch.Tensor, max_order: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Given a prediction u evaluated at data (t, x), returns du/dt and du/dx up to max_order, including ones
    as first column.

    Args:
        data (torch.Tensor): (t, x) locations of where to evaluate derivatives (n_samples x 2)
        prediction (torch.Tensor): the data u for which to evaluate the library (n_samples x 1)
        max_order (int): maximum order of derivatives to be calculated.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: time derivative and feature library ((n_samples, 1), (n_samples,  max_order + 1))
    """
    dy = grad(
        prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True
    )[0]
    time_deriv = dy[:, 0:1]


    if max_order == 0:
        du = torch.ones_like(time_deriv)
    else:
        du = torch.cat((torch.ones_like(time_deriv), dy[:, 1:2]), dim=1)
        if max_order > 1:
            for order in np.arange(1, max_order):
                du = torch.cat(
                    (
                        du,
                        grad(
                            du[:, order : order + 1],
                            data,
                            grad_outputs=torch.ones_like(prediction),
                            create_graph=True,
                        )[0][:, 1:2],
                    ),
                    dim=1,
                )

    return time_deriv, du


########## my function
def library_time_deriv(
    data: torch.Tensor, prediction: torch.Tensor, max_order: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    dy = grad(
        prediction, data, grad_outputs=torch.ones_like(prediction), create_graph=True
    )[0]
    time_deriv = dy[:, 0:1] 

    if max_order == 0:
        dt=torch.ones_like(time_deriv)#dt = torch.tensor([]) #torch.ones_like(time_deriv)
    else:
        dt = torch.cat((torch.ones_like(time_deriv), dy[:,0:1]), dim=1) #takes first derivative wrt time
        if max_order > 1:
            for order in np.arange(1, max_order):

                #takes derivative of previous derivative term wrt time
                dt = torch.cat(
                    (
                        dt,
                        grad(
                            dt[:, order : order + 1],
                            data,
                            grad_outputs=torch.ones_like(prediction),
                            create_graph=True,
                        )[0][:, 0:1],
                    ),
                    dim=1,
                )

    #print("dt")
    #print(dt)
    #print("wanted dt")
    #print(dt[:, 2:max_order+1])
    return time_deriv, dt[:, 2:max_order+1]
###########################    

# ========================= Actual library functions ========================
class Library1D(Library):
    def __init__(self, poly_order: int, diff_order: int, time_diff_order: int) -> None: #added time_diff_order as input
        """Calculates the temporal derivative a library/feature matrix consisting of
        1) polynomials up to order poly_order, i.e. u, u^2...
        2) derivatives up to order diff_order, i.e. u_x, u_xx
        3) cross terms of 1) and 2), i.e. $uu_x$, $u^2u_xx$

        Order of terms is derivative first, i.e. [$1, u_x, u, uu_x, u^2, ...$]

        Only works for 1D+1 data. Also works for multiple outputs but in that case doesn't calculate
        polynomial and derivative cross terms.

        Args:
            poly_order (int): maximum order of the polynomial in the library
            diff_order (int): maximum order of the differentials in the library
        """

        super().__init__()
        self.poly_order = poly_order
        self.diff_order = diff_order
        self.time_diff_order = time_diff_order #my addition

    def library(
        self, input: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[TensorList, TensorList]:
        """Compute the temporal derivative and library for the given prediction at locations given by data.
            Data should have t in first column, x in second.

        Args:
            input (Tuple[torch.Tensor, torch.Tensor]): A prediction u (n_samples, n_outputs) and spatiotemporal locations (n_samples, 2).

        Returns:
            Tuple[TensorList, TensorList]: The time derivatives [(n_samples, 1) x n_outputs] and the thetas [(n_samples, (poly_order + 1)(deriv_order + 1))]
            computed from the library and data.
        """
        prediction, data = input
        poly_list = []
        deriv_list = []
        time_deriv_list = []

        # Creating lists for all outputs
        for output in np.arange(prediction.shape[1]):
            time_deriv, du = library_deriv(
                data, prediction[:, output : output + 1], self.diff_order
            )
            
            dt = library_time_deriv(data, prediction[:, output : output + 1], self.time_diff_order)[1] #my addition
            
            u = library_poly(prediction[:, output : output + 1], self.poly_order)

            poly_list.append(u)
            #deriv_list.append(du)
            deriv_list.append(torch.cat((du, dt), 1)) #my addition
            #(1 means concantenate each row of du with the corresponding row of dt; 0 means add all of dt below du)
            time_deriv_list.append(time_deriv)


        samples = time_deriv_list[0].shape[0]
        total_terms = poly_list[0].shape[1] * deriv_list[0].shape[1]

        #my additions
        #print(deriv_list)
        #print("deriv_list[0]")
        #print(deriv_list[0])
        #print("deriv_list.shape[0]")
        #print(deriv_list[0].shape)
        #print(deriv_list[0].shape[1])
        #print("total_terms")
        #print(total_terms)


        # Calculating theta
        if len(poly_list) == 1:
            # If we have a single output, we simply calculate and flatten matrix product
            # between polynomials and derivatives to get library
            theta = torch.matmul(
                poly_list[0][:, :, None], deriv_list[0][:, None, :]
            ).view(samples, total_terms)
        else:
            theta_uv = reduce(
                (lambda x, y: (x[:, :, None] @ y[:, None, :]).view(samples, -1)),
                poly_list,
            )
            # calculate all unique combinations of derivatives
            theta_dudv = torch.cat(
                [
                    torch.matmul(du[:, :, None], dv[:, None, :]).view(samples, -1)[
                        :, 1:
                    ]
                    for du, dv in combinations(deriv_list, 2) #gets all combinations of 2 elements of deriv_list
                ],
                1,
            )
            theta = torch.cat([theta_uv, theta_dudv], dim=1)

        return time_deriv_list, [theta]


class Library2D(Library):
    def __init__(self, poly_order: int) -> None:
        """Create a 2D library up to given polynomial order with second order derivatives
         i.e. for poly_order=1: [$1, u_x, u_y, u_{xx}, u_{yy}, u_{xy}$]
        Args:
            poly_order (int): maximum order of the polynomial in the library
        """
        super().__init__()
        self.poly_order = poly_order

    def library(
        self, input: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[TensorList, TensorList]:
        """Compute the library for the given a prediction and data

        Args:
            input (Tuple[torch.Tensor, torch.Tensor]): A prediction and its data

        Returns:
            Tuple[TensorList, TensorList]: The time derivatives and the thetas
            computed from the library and data.
        """

        prediction, data = input
        # Polynomial

        u = torch.ones_like(prediction)
        for order in np.arange(1, self.poly_order + 1):
            u = torch.cat((u, u[:, order - 1 : order] * prediction), dim=1)

        # Gradients
        du = grad(
            prediction,
            data,
            grad_outputs=torch.ones_like(prediction),
            create_graph=True,
        )[0]
        u_t = du[:, 0:1]
        u_x = du[:, 1:2]
        u_y = du[:, 2:3]
        du2 = grad(
            u_x, data, grad_outputs=torch.ones_like(prediction), create_graph=True
        )[0]
        u_xx = du2[:, 1:2]
        u_xy = du2[:, 2:3]
        u_yy = grad(
            u_y, data, grad_outputs=torch.ones_like(prediction), create_graph=True
        )[0][:, 2:3]

        du = torch.cat((torch.ones_like(u_x), u_x, u_y, u_xx, u_yy, u_xy), dim=1)

        samples = du.shape[0]
        # Bringing it together
        theta = torch.matmul(u[:, :, None], du[:, None, :]).view(samples, -1)

        return [u_t], [theta]
