from models.ProtoLP import ProtoLP
from models.DataManifolds import ManiFolds
from models.groups import groups
CLASSIFIERS = {
    'baseline':ManiFolds,
	'protolp': ProtoLP,
    'group': groups,
}
