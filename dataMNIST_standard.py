import theano
import numpy
import time


class DataMNIST(object):
    def __init__(self, path, mbs, bs, rng, unlabled):
        self.path = path
        self.mbs = mbs
        self.bs = bs
        self.rng = rng
        self.unlabled = unlabled
        self.data = numpy.load(path)
        # print(self.data.files)
        # with open('mnist2.npz', 'wb') as f:
            # import numpy as np
            # np.savez(f, train_x=self.data['train'], train_y=self.data['train_labels'], test_x=self.data['test'], test_y=self.data['test_labels'])
        # with open('mnist3.npz', 'wb') as f:
            # import numpy as np
            # train_x = self.data['train_x'].T
            # train_y = self.data['train_y'].T.flatten()
            # test_x = self.data['test_x'].T
            # test_y = self.data['test_y'].T.flatten()
            # print('IMP -1', train_x.shape, train_y.shape, test_x.shape, test_y.shape, np.unique(train_y), np.unique(test_y))
            # valid_size = int(len(train_x) * .1)
            # valid_x, train_x = train_x[:valid_size], train_x[valid_size:]
            # valid_y, train_y = train_y[:valid_size], train_y[valid_size:]
            # train_x, valid_x, test_x = [x.astype(np.float64) / 255. for x in [train_x, valid_x, test_x]]
            # train_y, valid_y, test_y = [y.astype(np.int32) for y in [train_y, valid_y, test_y]]
            # np.savez(f, train_x=train_x, train_y=train_y, valid_x=valid_x, valid_y=valid_y, test_x=test_x, test_y=test_y)
        self.xdim = self.data['train_x'].shape[1]
        self.ydim = numpy.max(self.data['train_y'])+1

        self.offset = theano.shared(numpy.int32(0))
        self.begin = self.offset * self.mbs
        self.end = self.offset*self.mbs + self.mbs
        self._train_x = theano.shared(self.data['train_x'], name='train_x')
        self._train_y = theano.shared(self.data['train_y'], name='train_y')
        self._valid_x = theano.shared(self.data['valid_x'], name='valid_x')
        self._valid_y = theano.shared(self.data['valid_y'], name='valid_y')
        self._test_x = theano.shared(self.data['test_x'], name='test_x')
        self._test_y = theano.shared(self.data['test_y'], name='test_y')
        # Codes:
        # 0 -> same minibatch
        # 1 -> different minibatch
        # 2 -> validation set
        if unlabled == 0:
            self._natgrad = self._train_x[self.begin:self.end]
            self._natgrady = self._train_y[self.begin:self.end]
        elif unlabled == 1:
            self._natgrad = self._train_x[self.begin:self.end]
            self._natgrady = self._train_y[self.begin:self.end]
        elif unlabled == 2:
            self._natgrad = self._valid_x
            self._natgrady = self.valid_y

        self.eval_variables = [self._train_x,
                               self._train_y]
        self.n_valid_samples = self.data['valid_x'].shape[0]
        self.n_test_samples = self.data['test_x'].shape[0]

        self.n_batches = 50000 // self.bs
        self.nat_batches = self.n_batches

        if self.unlabled ==2:
            self.nat_batches = 10000 // self.mbs
        self.grad_perm = self.rng.permutation(self.n_batches)
        self.nat_perm = self.rng.permutation(self.nat_batches)
        self.variables = [self.train_x, self.train_y]
        self.train_variables = [
            self._train_x[self.offset*self.bs:
                          self.offset*self.bs+self.bs],
            self._train_y[self.offset*self.bs:
                          self.offset*self.bs+self.bs]]
        self.pos = -1
        self.nat_pos = -1

    def train_x(self, start, end):
        return self._train_x[
            self.offset*self.bs+start:self.offset*self.bs+end]

    def train_y(self, start, end):
        return self._train_y[
            self.offset*self.bs+start:self.offset*self.bs+end]

    def valid_x(self, start, end):
        return self._valid_x[start:end]

    def valid_y(self, start, end):
        return self._valid_y[start:end]

    def test_x(self, start, end):
        return self._test_x[start:end]

    def test_y(self, start, end):
        return self._test_y[start:end]

    def update_before_computing_gradients(self):
        self.pos = (self.pos + 1) % self.n_batches
        if self.pos % self.n_batches == 0:
            self.grad_perm = self.rng.permutation(self.n_batches)
        self.offset.set_value(self.grad_perm[self.pos])



    def update_before_computing_natural_gradients(self):
        if self.unlabled == 1:
            self.nat_pos = (self.nat_pos + 1) % self.nat_batches
            if self.nat_pos % self.nat_batches == 0:
                self.nat_perm = self.rng.permutation(self.nat_batches)
            self.offset.set_value(self.nat_perm[self.nat_pos])

    def update_before_evaluation(self):
        if self.unlabled == 1:
            self.offset.set_value(self.grad_perm[self.pos])
