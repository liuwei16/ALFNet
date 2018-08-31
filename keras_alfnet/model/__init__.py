# def creat_model(opt):
#     if opt.step == 1:
#         from .model_1step import Model_1step
#         model = Model_1step()
#     elif opt.step == 2:
#         from .model_2step import Model_2step
#         model = Model_2step()
#     elif opt.step == 3:
#         from .model_3step import Model_3step
#         model = Model_3step()
#     else:
#         raise NotImplementedError('Not implement {} or more steps'.format(opt.steps))
#     model.initialize(opt)
#     print ('model {} was created'.format(model.name()))

