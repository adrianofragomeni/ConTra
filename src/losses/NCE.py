import torch as th    

class Lossfunction(th.nn.Module):
    def __init__(self, device, lambda_1, lambda_2, lambda_3, lambda_4, temperature):
        super(Lossfunction, self).__init__()
        
        self.temperature=temperature
        self.lambda_1=lambda_1
        self.lambda_2=lambda_2
        self.lambda_3=lambda_3
        self.lambda_4=lambda_4
        self.device=device

    def nce_loss(self,text_embed,video_embed):
        
        similarity_matrix=th.matmul(text_embed, video_embed.t())/ self.temperature
        x = similarity_matrix.t()
        x = x.view(similarity_matrix.shape[0], similarity_matrix.shape[0], -1)
        nominator = x * th.eye(x.shape[0])[:,:,None].to(self.device)
        nominator = nominator.sum(dim=1)

        denominator = th.cat((x, x.permute(1,0,2)), dim=1).view(x.shape[0], -1)
        denominator = th.logsumexp(denominator, dim=1)
        
        return self.lambda_1*th.mean(denominator - nominator)        

    def uniformity_loss(self,set_, t=2):
        
        return self.lambda_2*th.pdist(set_, p=2).pow(2).mul(-t).exp().mean().log()

    def neighbouring_text_loss(self,video, text, negatives):
        
        nominator=th.unsqueeze(th.diag(th.matmul(text, video.t())/ self.temperature),1)
        neg=th.unsqueeze(th.diag(th.matmul(negatives, video.t())/ self.temperature),1)
        denominator=th.cat([nominator,neg],1)
        denominator = th.logsumexp(denominator, dim=1)
        
        return self.lambda_3*th.mean(denominator - nominator)
        
    def neighbouring_video_loss(self,video, text, negatives):
        nominator=th.unsqueeze(th.diag(th.matmul(video, text.t())/ self.temperature),1)
        neg=th.unsqueeze(th.diag(th.matmul(negatives, text.t())/ self.temperature),1)
        denominator=th.cat([nominator,neg],1)
        denominator = th.logsumexp(denominator, dim=1)
        return self.lambda_4*th.mean(denominator - nominator)
    

    def forward(self, video_embed, text_embed, negative_embed_text,negative_embed_video, mask):
        
        if self.lambda_1:
            nce_loss_value=self.nce_loss(text_embed,video_embed)
        else:
            nce_loss_value=th.tensor(0,requires_grad=False)
        
        if self.lambda_2:
            uniformity_loss_value=self.uniformity_loss(th.cat([video_embed, text_embed]))
        else:
            uniformity_loss_value=th.tensor(0,requires_grad=False)
            
        if self.lambda_3:
            neighbouring_text_loss_value=self.neighbouring_text_loss(video_embed[mask,:], text_embed[mask,:], negative_embed_text)
        else:
            neighbouring_text_loss_value=th.tensor(0,requires_grad=False)
            
        if self.lambda_4:
            neighbouring_video_loss_value=self.neighbouring_video_loss(video_embed[mask,:], text_embed[mask,:],negative_embed_video)
        else:
            neighbouring_video_loss_value=th.tensor(0,requires_grad=False)
            
            
        loss=nce_loss_value + uniformity_loss_value + neighbouring_text_loss_value + neighbouring_video_loss_value
        
        loss_dict={"Inter_loss":nce_loss_value.item(), "Uniformity_loss":uniformity_loss_value.item() ,
                   "Neighbouring_video_loss":neighbouring_video_loss_value.item(),"Neighbouring_text_loss":neighbouring_text_loss_value.item() }

        return loss, loss_dict
