import torch

def main():
    import my_inception_v3
    model = my_inception_v3.inception_v3()
    model = model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu")).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = torch.jit.script(model)
    model.cuda()
    model.eval()
    model.save("Inception_v3.pt")

    from spyNet import Network
    arguments_strModel = "sintel-final"
    net = Network()
    net.load_state_dict(
        {
            strKey.replace("module", "net"): tenWeight
            for strKey, tenWeight in torch.hub.load_state_dict_from_url(
                url="http://content.sniklaus.com/github/pytorch-spynet/network-"
                + arguments_strModel
                + ".pytorch",
                file_name="spynet-" + arguments_strModel,
            ).items()
        }
    )
    net = net.cuda().eval()
    model = torch.jit.script(net)
    for p in model.parameters():
        p.requires_grad_(False)
    model.cuda()
    model.eval()
    model.save("spynet.pt")

if __name__ == "__main__":
    sys.exit(main())