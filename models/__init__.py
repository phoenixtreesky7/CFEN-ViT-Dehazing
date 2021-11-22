import logging

def create_model(opt):
	model = None
	if opt.model == 'vit': 
		print('the model is vit')
		from .mgvit_model import MGVIT
		model = MGVIT()
	elif opt.model == 'dec_vit':
		from .model_iid_dehazing import DECHLGVIT
		model = DECHLGVIT()
	elif opt.model == 'decr_vit':
		from .model_iidr_dehazing import DECRHLGVIT
		model = DECRHLGVIT()
	elif opt.model == 'decs_vit':
		from .model_iids_dehazing import DECSHLGVIT
		model = DECSHLGVIT()
	elif opt.model == 'decn_vit':
		from .model_iidn_dehazing import DECNHLGVIT
		model = DECNHLGVIT()
	
	elif opt.model == 'test':
		from .test_model import TestModel
		model = TestModel()
	else:
		raise NotImplementedError('model [%s] not implemented.' % opt.model)
	model.initialize(opt)
	logging.info("model [%s] was created" % (model.name()))
	return model
