"""Configuration File"""

TARGET_COLS = 'y'

CAT_COLS = ['job',
            'marital',
            'education',
            'default',
            'housing',
            'loan',
            'contact',
            'month',
            'day_of_week',
            'poutcome']

NUM_COLS = ['age',
            'campaign',
            'pdays',
            'previous',
            'emp.var.rate',
            'cons.price.idx',
            'cons.conf.idx',
            'euribor3m',
            'nr.employed']

DROP_COLS = ["duration"]
