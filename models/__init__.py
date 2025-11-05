from models.ProtoLP import ProtoLP
from models.DataManifolds import ManiFolds
from models.groups import groups
from models.groupsiter import groupsiter
CLASSIFIERS = {
    'baseline':ManiFolds,
	'protolp': ProtoLP,
    'group': groups,
    'groupiter': groupsiter
}
