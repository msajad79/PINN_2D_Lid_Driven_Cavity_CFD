import time
import torch
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch import nn, optim
from loss import VSFLoss
from dataprocess import DataLoader, LabeledDataset, PhysicsDataset
from model import FCN

# setup program
writer = SummaryWriter()
device = torch.device("cuda")

# data process
def data_process():
    train_lbl_ds = LabeledDataset(
        df=pd.read_csv("./data/Re1_100i100.csv"),
        device=device
    )
    train_phy_ds = PhysicsDataset(
        step_x=10,
        step_y=10,
        device=device
    )
    train_lbl_dl = DataLoader(train_lbl_ds, batch_size=len(train_lbl_ds), shuffle=True)
    train_phy_dl = DataLoader(train_phy_ds, batch_size=len(train_phy_ds), shuffle=True)

    test_lbl_ds = LabeledDataset(
        df=pd.read_csv("./data/Re1_100i100.csv"),
        device=device
    )
    test_phy_ds = PhysicsDataset(
        step_x=50,
        step_y=50,
        device=device
    )
    test_lbl_dl = DataLoader(test_lbl_ds, batch_size=len(test_lbl_ds), shuffle=False)
    test_phy_dl = DataLoader(test_phy_ds, batch_size=len(test_phy_ds), shuffle=False)
    return train_lbl_ds, train_lbl_dl, train_phy_ds, train_phy_dl, test_lbl_ds, test_lbl_dl, test_phy_ds, test_phy_dl
train_lbl_ds, train_lbl_dl, train_phy_ds, train_phy_dl, test_lbl_ds, test_lbl_dl, test_phy_ds, test_phy_dl = data_process()

# setup model
model = FCN(N_INPUT=2, N_OUTPUT=2, N_HIDDEN=32, N_LAYERS=4).to(device)
#model = torch.load("./models/model_28_8_9_1_sajad.pt")
criterion = VSFLoss(Re=1.0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=350, gamma=.9)
epochs = 120000


epochs = 120000
MIN_LOSS_GVEQ = 0.01
for epoch in (pbar := tqdm(range(epochs))):

    inp_train_lbl, target_train_lbl = next(iter(train_lbl_dl))
    inp_train_phy = next(iter(train_phy_dl))

    # forward
    out_lbl = model(inp_train_lbl[:,[0,1]]) # just send x,y for model
    out_phy = model(inp_train_phy[:,[0,1]]) # just send x,y for model
    
    # loss
    loss = criterion(inp_train_lbl, target_train_lbl[:,[0,1]], out_lbl, inp_train_phy, out_phy) # just send u, v for target label
    
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # pbar log
    pbar.set_description(f"loss : {loss.item()}")

    # TensorBoard
    writer.add_scalars("loss/train", {
            "loss lbl u":criterion.loss_lbl_u, 
            "loss lbl v":criterion.loss_lbl_v, 
            "BC left":criterion.loss_BC_left,
            "BC right":criterion.loss_BC_right,
            "BC down":criterion.loss_BC_down,
            "BC up":criterion.loss_BC_up,
            "loss GvEq":criterion.loss_GvEq
        }, epoch)
    writer.add_scalars("losslambda/train", {
            "loss lbl u":criterion.loss_lbl_u * criterion.lambda_lbl_u, 
            "loss lbl v":criterion.loss_lbl_v * criterion.lambda_lbl_v, 
            "BC left":criterion.loss_BC_left * criterion.lambda_BC_wall_noslip,
            "BC right":criterion.loss_BC_right * criterion.lambda_BC_wall_noslip,
            "BC down":criterion.loss_BC_down * criterion.lambda_BC_wall_noslip,
            "BC up":criterion.loss_BC_up * criterion.lambda_BC_wall_move,
            "loss GvEq":criterion.loss_GvEq * criterion.lambda_GvEq
        }, epoch)
    
    # validation
    if epoch % 200 == 0:
        model.eval()

        # predict physics test data
        inp_test_phy = next(iter(test_phy_dl))
        out_phy_test = model(inp_test_phy[:, [0,1]]) # just send x,y for model

        # params test
        x_test = inp_test_phy[:,0].cpu().detach()
        y_test = inp_test_phy[:,1].cpu().detach()
        u_test = out_phy_test[:,0].cpu().detach()
        v_test = out_phy_test[:,1].cpu().detach()
        # Concatenate the two check tensors along the columns to create a 2D tensor
        combined_check = torch.stack((x_test, y_test), dim=1)

        # Sort the combined_check tensor along dim=0 (rows)
        sorted_indices = torch.argsort(combined_check[:, 1])  # Sort based on sec_check first
        sorted_indices = sorted_indices[torch.argsort(combined_check[sorted_indices][:, 0], stable=True)]  # Then sort based on first_check

        # Use the sorted indices to rearrange the target_tensor
        u_test_s = u_test[sorted_indices]
        v_test_s = v_test[sorted_indices]


        # vector figure test
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.quiver(
            x_test,
            y_test,
            u_test,
            v_test,
        )
        ax.scatter(inp_train_lbl[:,0].cpu(), inp_train_lbl[:,1].cpu())
        writer.add_figure("vector/train", fig, epoch)

        # cantour U figure
        fig, ax = plt.subplots(1, figsize=(10, 10))
        cp = ax.contour(x_test.unique(), y_test.unique(), u_test_s.view(test_phy_ds.step_x,test_phy_ds.step_y), colors='black', linestyles='dashed', linewidths=1)
        ax.clabel(cp, inline=1, fontsize=10)
        cp = ax.contourf(x_test.unique(), y_test.unique(), u_test_s.view(test_phy_ds.step_x,test_phy_ds.step_y))
        cb = fig.colorbar(cp)
        writer.add_figure("cantourU/train", fig, epoch)
        plt.close()

        # cantour V figure
        fig, ax = plt.subplots(1, figsize=(10, 10))
        cp = ax.contour(x_test.unique(), y_test.unique(), v_test_s.view(test_phy_ds.step_x,test_phy_ds.step_y), colors='black', linestyles='dashed', linewidths=1)
        ax.clabel(cp, inline=1, fontsize=10)
        cp = ax.contourf(x_test.unique(), y_test.unique(), v_test_s.view(test_phy_ds.step_x,test_phy_ds.step_y))
        cb = fig.colorbar(cp)
        writer.add_figure("cantourV/train", fig, epoch)
        plt.close()
    scheduler.step()

# train
t_mean = [0.0]*5
for epoch in (pbar := tqdm(range(epochs))):
    optimizer.zero_grad()
    ts = []
    for xyt in train_dl:
        ts.append(time.time())
        uv = model(xyt)
        ts.append(time.time())
        loss = criterion(uv, xyt)
        ts.append(time.time())
        loss.backward()
        ts.append(time.time())
    optimizer.step()
    ts.append(time.time())
    pbar.set_description(f"loss : {loss.item()}")
    writer.add_scalar("Loss/train", loss, epoch)        
    writer.add_scalars("LossX/train", {
                "loss BC noslip":criterion.loss_BC_wall_noslip, 
                "loss BC move":criterion.loss_BC_wall_move, 
                "loss gveq":criterion.loss_GvEq
            }, epoch) 
    writer.add_scalars("lambda/train", {
                "lambda BC noslip":criterion.lambda_BC_wall_noslip, 
                "lambda BC move":criterion.lambda_BC_wall_move,
            }, epoch)
    ts.append(time.time())
    for i, (t1, t2) in enumerate(zip(ts[:-1],ts[1:])):
        t_mean[i] = (t_mean[i]*epoch + t2-t1)/(epoch+1)
    if epoch % 100 == 0:
        model.eval()
        uv_test = model(test_ds.data)

        # vector figure
        fig, ax = plt.subplots(1, figsize=(10, 10))
        mask = test_ds.data[:, 2] == 0.0
        ax.quiver(
            test_ds.data[mask][:, 0].cpu().detach(),
            test_ds.data[mask][:, 1].cpu().detach(),
            uv_test[mask][:, 0].cpu().detach(),
            uv_test[mask][:, 1].cpu().detach(),
        )
        writer.add_figure("vector/train", fig, epoch)

        # cantour U figure
        fig, ax = plt.subplots(1, figsize=(10, 10))
        cp = ax.contour(torch.linspace(0,1,test_ds.step_x), torch.linspace(0,1,test_ds.step_y), uv_test[:,0].view(test_ds.step_x,test_ds.step_y).cpu().detach(), colors='black', linestyles='dashed', linewidths=1)
        ax.clabel(cp, inline=1, fontsize=10)
        cp = ax.contourf(torch.linspace(0,1,test_ds.step_x), torch.linspace(0,1,test_ds.step_y), uv_test[:,0].view(test_ds.step_x,test_ds.step_y).cpu().detach())
        cb = fig.colorbar(cp)
        writer.add_figure("cantourU/train", fig, epoch)
        plt.close()

        # cantour V figure
        fig, ax = plt.subplots(1, figsize=(10, 10))
        cp = ax.contour(torch.linspace(0,1,test_ds.step_x), torch.linspace(0,1,test_ds.step_y), uv_test[:,1].view(test_ds.step_x,test_ds.step_y).cpu().detach(), colors='black', linestyles='dashed', linewidths=1)
        ax.clabel(cp, inline=1, fontsize=10)
        cp = ax.contourf(torch.linspace(0,1,test_ds.step_x), torch.linspace(0,1,test_ds.step_y), uv_test[:,1].view(test_ds.step_x,test_ds.step_y).cpu().detach())
        cb = fig.colorbar(cp)
        writer.add_figure("cantourV/train", fig, epoch)
        plt.close()

        #contour loss GvEq
        fig, ax = plt.subplots(1, figsize=(10, 10))
        cp = ax.contour(torch.linspace(0,1,train_ds.step_x), torch.linspace(0,1,train_ds.step_y), criterion.loss_GvEqs.cpu().detach().reshape(train_ds.step_x,train_ds.step_y), colors='black', linestyles='dashed', linewidths=1)
        ax.clabel(cp, inline=1, fontsize=10)
        cp = ax.contourf(torch.linspace(0,1,train_ds.step_x), torch.linspace(0,1,train_ds.step_y), criterion.loss_GvEqs.cpu().detach().reshape(train_ds.step_x,train_ds.step_y))
        cb = fig.colorbar(cp)
        writer.add_figure("cantourGVEqsloss/train", fig, epoch)
        plt.close()
        if epoch < 1501:
            train_ds, test_ds, train_dl, test_dl = data_process(train_ds.step_x+1)
