 model = get_model()
                feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])