"""Example runner"""
from src.train.trainer import train_closed_form

if __name__=='__main__':
    train_closed_form('/home/lathika/ReTM_Workspace/Recordings/Splited_data/Channels_7/train', device='cuda')