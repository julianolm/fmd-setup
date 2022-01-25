def load_data(data_dir='/content/image', seed=42, train_rate=0.5):
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    dataset_train = datasets.ImageFolder(data_dir, transform=transform_train)
    dataset_test  = datasets.ImageFolder(data_dir, transform=transform_test)

    num_train = int(len(dataset_train) * train_rate)
    num_test = len(dataset_train) - num_train

    trainset = torch.utils.data.random_split(dataset_train, [num_train,num_test], generator=torch.Generator().manual_seed(seed))[0]
    testset  = torch.utils.data.random_split(dataset_test,  [num_train,num_test], generator=torch.Generator().manual_seed(seed))[1]

    return trainset, testset
    
    
trainset, testset = load_data(seed=k, train_rate=0.5)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
