from typing import Dict


class Label:
    shape_to_tag = dict(rectangle='box')

    def __init__(self, class_id: int, infos: Dict[str, str]) -> None:
        self.cid = class_id
        self.name = infos['name']
        self.shape = infos['type']
        self.tag = self.shape_to_tag.get(self.shape, self.shape)

    def __repr__(self) -> str:
        return f"{self.name} cid={self.cid} shape={self.shape}"
