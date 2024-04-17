import torch
from torch import nn

class VSFLoss(nn.Module):

    def __init__(self, Re:float):
        super(VSFLoss, self).__init__()
        self.lambda_BC_wall_move = torch.tensor(1.0e1)
        self.lambda_BC_wall_noslip = torch.tensor(1.0e1)
        self.lambda_lbl_u = torch.tensor(1.0)
        self.lambda_lbl_v = torch.tensor(1.0e1)
        self.lambda_GvEq = torch.tensor(1.0e2)

    def learning_rate_annealing(self, uv, xyt, alpha=.9):
        dL_GvEq_dt = torch.autograd.grad(self.loss_GvEqs, xyt, torch.ones_like(self.loss_GvEqs), create_graph=True)[0]
        dL_BC_wall_moves_dt = torch.autograd.grad(self.loss_BC_wall_moves, xyt, torch.ones_like(self.loss_BC_wall_moves), create_graph=True)[0]
        dL_BC_wall_noslips_dt = torch.autograd.grad(self.loss_BC_wall_noslips, xyt, torch.ones_like(self.loss_BC_wall_noslips), create_graph=True)[0]

        lambda_hat_BC_wall_moves = torch.max(torch.abs(dL_GvEq_dt))/torch.mean(torch.abs(dL_BC_wall_moves_dt))
        lambda_hat_BC_wall_noslip = torch.max(torch.abs(dL_GvEq_dt))/torch.mean(torch.abs(dL_BC_wall_noslips_dt))
        self.lambda_BC_wall_move = torch.min(torch.tensor([alpha*self.lambda_BC_wall_move + (1-alpha)*lambda_hat_BC_wall_moves, 1e12]))
        self.lambda_BC_wall_noslip = torch.min(torch.tensor([alpha*self.lambda_BC_wall_noslip + (1-alpha)*lambda_hat_BC_wall_noslip, 1e12]))

    def my_balanced(self, alpha=.9):
        self.lambda_BC_wall_move = torch.abs(self.loss_BC_wall_move)/(torch.abs(self.loss_BC_wall_move)+torch.abs(self.loss_BC_wall_noslip)+torch.abs(self.loss_GvEq))
        self.lambda_BC_wall_noslip = torch.abs(self.loss_BC_wall_noslip)/(torch.abs(self.loss_BC_wall_move)+torch.abs(self.loss_BC_wall_noslip)+torch.abs(self.loss_GvEq))*10.0
        self.lambda_GvEq = torch.abs(self.loss_GvEq)/(torch.abs(self.loss_BC_wall_move)+torch.abs(self.loss_BC_wall_noslip)+torch.abs(self.loss_GvEq))

    def forward(self, inp_lbl:torch.tensor, target_lbl:torch.tensor, out_lbl:torch.tensor, inp_phy:torch.tensor, out_phy:torch.tensor):
        # calculate labled data loss
        self.loss_lbl_u = torch.nn.functional.mse_loss(target_lbl[:,0], out_lbl[:,0])
        self.loss_lbl_v = torch.nn.functional.mse_loss(target_lbl[:,1], out_lbl[:,1])


        # calculate physics data loss
        # create mask
        mask_phy_left = inp_phy[:,0] == 0.0
        mask_phy_right = inp_phy[:,0] == 1.0
        mask_phy_down = inp_phy[:,1] == 0.0
        mask_phy_up = inp_phy[:,1] == 1.0

        # calculate grads
        dudx, dudy = torch.autograd.grad(out_phy[:,[0]], inp_phy, torch.ones_like(out_phy[:,[0]]), create_graph=True)[0].unbind(dim=1)
        dvdx, dvdy = torch.autograd.grad(out_phy[:,[1]], inp_phy, torch.ones_like(out_phy[:,[1]]), create_graph=True)[0].unbind(dim=1)
        #d2udxx, d2udxy = torch.autograd.grad(dudx, inp_phy, torch.ones_like(dudx), create_graph=True)[0].unbind(dim=1)
        #_     , d2udyy = torch.autograd.grad(dudy, inp_phy, torch.ones_like(dudy), create_graph=True)[0].unbind(dim=1)
        #d2vdxx, d2vdxy = torch.autograd.grad(dvdx, inp_phy, torch.ones_like(dvdx), create_graph=True)[0].unbind(dim=1)
        #_     , d2vdyy = torch.autograd.grad(dvdy, inp_phy, torch.ones_like(dvdy), create_graph=True)[0].unbind(dim=1)

        w = dvdx - dudy # calculate vorticity
        dwdx, dwdy = torch.autograd.grad(w, inp_phy, torch.ones_like(w), create_graph=True)[0].unbind(dim=1)
        d2wdxx, d2wdxy = torch.autograd.grad(dwdx, inp_phy, torch.ones_like(dwdx), create_graph=True)[0].unbind(dim=1)
        _     , d2wdyy = torch.autograd.grad(dwdy, inp_phy, torch.ones_like(dwdy), create_graph=True)[0].unbind(dim=1)
        
        # BC loss
        self.loss_BC_left = torch.nn.functional.mse_loss(torch.zeros_like( out_phy[mask_phy_left]), out_phy[mask_phy_left] )
        self.loss_BC_right = torch.nn.functional.mse_loss(torch.zeros_like( out_phy[mask_phy_right]), out_phy[mask_phy_right] )
        self.loss_BC_down = torch.nn.functional.mse_loss(torch.zeros_like( out_phy[mask_phy_down]), out_phy[mask_phy_down] )
        self.loss_BC_up = torch.nn.functional.mse_loss(
            torch.concat(( torch.ones(out_phy[mask_phy_up].shape[0], 1), torch.zeros( out_phy[mask_phy_up].shape[0], 1) ),axis=1).to(out_phy.device),
            out_phy[mask_phy_up]
        )

        # Governing Eq loss
        Re = 1.0
        self.loss_GvEq = torch.mean(( out_phy[:,0]*dwdx + out_phy[:,1]*dwdy - 1.0/Re * (d2wdxx + d2wdyy) )**2.0)

        # TODO:balnce loss

        # store important variable
        self.w = w
        
        return (
            self.loss_lbl_u * self.lambda_lbl_u +
            self.loss_lbl_v * self.lambda_lbl_v +
            self.loss_BC_left * self.lambda_BC_wall_noslip + 
            self.loss_BC_right * self.lambda_BC_wall_noslip + 
            self.loss_BC_down * self.lambda_BC_wall_noslip + 
            self.loss_BC_up * self.lambda_BC_wall_move + 
            self.loss_GvEq * self.lambda_GvEq
        )