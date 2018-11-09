import logging
import argparse
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import ndf
import dataset
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder


def parse_arg():
	logging.basicConfig(
		level=logging.WARNING,
		format="[%(asctime)s]: %(levelname)s: %(message)s"
	)
	parser = argparse.ArgumentParser(description='train.py')
	parser.add_argument('-dataset', choices=['mnist', 'adult', 'letter', 'yeast', 'credit'], default='mnist')
	parser.add_argument('-batch_size', type=int, default=128)

	parser.add_argument('-feat_dropout', type=float, default=0.3)
	parser.add_argument('-test_batch', type=int, default=29878)
	parser.add_argument('-n_tree', type=int, default=5)
	parser.add_argument('-tree_depth', type=int, default=3)
	parser.add_argument('-n_class', type=int, default=10)
	parser.add_argument('-tree_feature_rate', type=float, default=0.5)

	parser.add_argument('-lr', type=float, default=0.001, help="sgd: 10, adam: 0.001")
	parser.add_argument('-gpuid', type=int, default=-1)
	parser.add_argument('-jointly_training', action='store_true', default=False)
	parser.add_argument('-epochs', type=int, default=10)
	parser.add_argument('-report_every', type=int, default=10)

	opt = parser.parse_args()
	return opt


def prepare_db(opt):
	print("Use %s dataset" % (opt.dataset))

	if opt.dataset == 'mnist':
		train_dataset = torchvision.datasets.MNIST('./data/mnist', train=True, download=True,
												   transform=torchvision.transforms.Compose([
													   torchvision.transforms.ToTensor(),
													   torchvision.transforms.Normalize((0.1307,), (0.3081,))
												   ]))

		eval_dataset = torchvision.datasets.MNIST('./data/mnist', train=False, download=True,
												  transform=torchvision.transforms.Compose([
													  torchvision.transforms.ToTensor(),
													  torchvision.transforms.Normalize((0.1307,), (0.3081,))
												  ]))
		return {'train': train_dataset, 'eval': eval_dataset}

	elif opt.dataset == 'adult':
		train_dataset = dataset.UCIAdult('./data/uci_adult', train=True)
		eval_dataset = dataset.UCIAdult('./data/uci_adult', train=False)
		#print(train_dataset)
		return {'train': train_dataset, 'eval': eval_dataset}

	elif opt.dataset == 'letter':
		train_dataset = dataset.UCILetter('./data/uci_letter', train=True)
		eval_dataset = dataset.UCILetter('./data/uci_letter', train=False)
		return {'train': train_dataset, 'eval': eval_dataset}

	elif opt.dataset == 'yeast':
		train_dataset = dataset.UCIYeast('./data/uci_yeast', train=True)
		eval_dataset = dataset.UCIYeast('./data/uci_yeast', train=False)
		return {'train': train_dataset, 'eval': eval_dataset}

	elif opt.dataset == 'credit':
		train_dataset = dataset.UCICredit('/home/ubuntu/amberhsia/Neural-Decision-Forests-master/data/q4credit.csv',
										  train=True)
		eval_dataset = dataset.UCICredit('/home/ubuntu/amberhsia/Neural-Decision-Forests-master/data/q4credit.csv',
										 train=False)
		return {'train': train_dataset, 'eval': eval_dataset}

	else:
		raise NotImplementedError


def prepare_model(opt):
	if opt.dataset == 'mnist':
		feat_layer = ndf.MNISTFeatureLayer(opt.feat_dropout)
	elif opt.dataset == 'adult':
		feat_layer = ndf.UCIAdultFeatureLayer(opt.feat_dropout)
	elif opt.dataset == 'letter':
		feat_layer = ndf.UCILetterFeatureLayer(opt.feat_dropout)
	elif opt.dataset == 'yeast':
		feat_layer = ndf.UCIYeastFeatureLayer(opt.feat_dropout)
	elif opt.dataset == 'credit':
		feat_layer = ndf.UCICreditFeatureLayer(opt.feat_dropout)
	else:
		raise NotImplementedError

	forest = ndf.Forest(n_tree=opt.n_tree, tree_depth=opt.tree_depth, n_in_feature=feat_layer.get_out_feature_size(),
						tree_feature_rate=opt.tree_feature_rate, n_class=opt.n_class,
						jointly_training=opt.jointly_training)
	model = ndf.NeuralDecisionForest(feat_layer, forest)

	if opt.cuda:
		model = model.cuda()
	else:
		model = model.cpu()

	return model


def prepare_optim(model, opt):
	params = [p for p in model.parameters() if p.requires_grad]
	return torch.optim.Adam(params, lr=opt.lr, weight_decay=1e-5)


def train(model, optim, db, opt):
	for epoch in range(1, opt.epochs + 1):
		# Update \Pi
		if not opt.jointly_training:  # if joint_training == False
			print("Epcho %d : Two Stage Learning - Update PI" % (epoch))

			# initialize an identity matrix for one-hot labels
			cls_onehot = torch.eye(opt.n_class)

			feat_batches = []
			target_batches = []
			#print('PRINT-DATA_TRAIN', db['train'])
			train_loader = torch.utils.data.DataLoader(db['train'], batch_size=opt.batch_size, shuffle=True)

			# data preprocessing for batch-size
			for batch_idx, (data, target) in enumerate(train_loader):
				# print('TRAINING_DATA_CHECK`')
				# print(data, target)
				if opt.cuda:
					data, target, cls_onehot = data.cuda(), target.cuda(), cls_onehot.cuda()
				# print('data: ', data)
				# print('target: ', target)

				# Get feats: input features for one batch
				data = Variable(data, volatile=True)
				feats = model.feature_layer(data)
				feats = feats.view(feats.size()[0], -1)
				feat_batches.append(feats)
				# print(feat_batches)

				# Get labels: one-hot labels for one batch
				mid = []
				for item in target:
					val_arr = cls_onehot[item]
					mid.append(val_arr)
				# print('one_batch: ', mid)
				tensor_mid = torch.stack(mid)  # convert a list of tensors to a tensor
				target_batches.append(tensor_mid)

			# Update \Pi for each tree
			for tree in model.forest.trees:  # for each tree
				mu_batches = []  							# mu = path routing prob.
				for feats in feat_batches:  # for each input
					mu = tree(feats)  						# [batch_size, n_leaf= # prob. belonging to each leaves]
					mu_batches.append(mu)

				for _ in range(20):
					new_pi = torch.zeros((tree.n_leaf, tree.n_class))  # [n_leaf,n_class]
					if opt.cuda:
						new_pi = new_pi.cuda()

					# load mu and label
					for mu, target in zip(mu_batches, target_batches):
						pi = tree.get_pi()  # [n_leaf,n_class]  equally distributed
						prob = tree.cal_prob(mu, pi)  # P_t: [batch_size,n_class]

						# Variable to Tensor
						pi = pi.data
						prob = prob.data
						mu = mu.data

						_target = target.unsqueeze(1)  # add one-dim: [batch_size,1,n_class]
						_pi = pi.unsqueeze(0)  # [1,n_leaf,n_class]
						_mu = mu.unsqueeze(2)  # [batch_size,n_leaf,1]
						# Clamp all elements in input into the range [min, max] and return a resulting Tensor.
						_prob = torch.clamp(prob.unsqueeze(1), min=1e-6, max=1.)  # [batch_size,1,n_class]

						# Loss = （y*pi）* mu / p_t
						_new_pi = torch.mul(torch.mul(_target, _pi), _mu) / _prob  # [batch_size,n_leaf,n_class]
						new_pi += torch.sum(_new_pi, dim=0)     # normalize

					# test
					# import numpy as np
					# if np.any(np.isnan(new_pi.cpu().numpy())):
					#    print(new_pi)
					# test
					new_pi = F.softmax(Variable(new_pi), dim=1).data
					tree.update_pi(new_pi)

		# Update \Theta
		model.train()

		train_loader = torch.utils.data.DataLoader(db['train'], batch_size=opt.batch_size, shuffle=True)
		for batch_idx, (data, target) in enumerate(train_loader):
			if opt.cuda:
				data, target = data.cuda(), target.cuda()
			data, target = Variable(data), Variable(target)

			# set the gradients to zero before starting to do backpropragation
			# clear out the gradients of all Variables
			optim.zero_grad()
			output = model(data)
			#print(output.shape)
			#print('Model_output: ', output)  # [1000, 2]

			# print('target: ', target.shape, target) # 0-1 LABEL without one-hot  [1000]
			# print('--------------------------')
			# print(torch.log(output))

			loss = F.nll_loss(torch.log(output), target.long())
			loss.backward()	# back-propagation
			# torch.nn.utils.clip_grad_norm([ p for p in model.par  ameters() if p.requires_grad],
			#                              max_norm=5)
			optim.step()

			if batch_idx % opt.report_every == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, batch_idx * len(data), len(train_loader.dataset),
						   100. * batch_idx / len(train_loader), loss.data[0]))

		# Eval
		model.eval()
		test_loss = 0
		correct = 0
		res = []
		auc = 0
		y_true = []
		y_pred =[]

		test_loader = torch.utils.data.DataLoader(db['eval'], batch_size=opt.batch_size, shuffle=True)
		for data, target in test_loader:
			if opt.cuda:
				data, target = data.cuda(), target.cuda()
			data, target = Variable(data, volatile=True), Variable(target).long()

			output = model(data)    # prob
			test_loss += F.nll_loss(torch.log(output), target, size_average=False).item()  # sum up batch loss
			
			pred = output.data.cpu().numpy()    # tensor to numpy
            
            # one-hot for y-true
			integer_encoded = target.data.cpu().numpy()
			onehot_encoder = OneHotEncoder(sparse=False)
			integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
			onehot_true = onehot_encoder.fit_transform(integer_encoded)
            
			#print('pred: ', type(pred), pred.shape) # ndarray (1000, 2)
			#print('true: ', type(onehot_true), onehot_true.shape)
            
			#
			#print('TEST-batch auc: ', auc)	
            
			y_pred = np.append(y_pred, pred)
			y_true = np.append(y_true, onehot_true)
		print(y_pred.shape, y_true.shape)
                  
		auc = roc_auc_score(y_true, y_pred)
		print('whole auc evaluate: ', auc)

		test_loss /= len(test_loader.dataset)
		print('\nTest set: Average loss: {:.4f}'.format(test_loss))


def main():
	opt = parse_arg()

	# GPU
	opt.cuda = opt.gpuid >= 0
	if opt.gpuid >= 0:
		torch.cuda.set_device(opt.gpuid)
	else:
		print("WARNING: RUN WITHOUT GPU")

	db = prepare_db(opt)
	#print('DATA_CHECK')
	#print(db)
	model = prepare_model(opt)
	optim = prepare_optim(model, opt)
	train(model, optim, db, opt)


if __name__ == '__main__':
	main()
