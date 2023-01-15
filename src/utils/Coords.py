from typing import Dict, List, Tuple, Union


class Coords:
    def __init__(self, obj: Dict[str, str], tag: str) -> None:
        if tag == "box":
            self._from_box(obj)
        elif tag == "ellipse":
            self._from_ellipse(obj)
        else:
            raise ValueError(f"Unknown tag `{tag}`.")

    def _extract_attributes(self, obj: Dict[str, str], attributes: List[str]) -> Tuple[float, float, float, float]:
        return (float(obj[attr]) for attr in attributes)

    def _from_box(self, obj: Dict[str, str], attrs: List[str] = ['@xtl', '@xbr', '@ytl', '@ybr']) -> None:
        xmin, xmax, ymin, ymax = self._extract_attributes(obj, attrs)

        self.cx, self.cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        self.w, self.h = xmax - xmin, ymax - ymin

    def _from_ellipse(self, obj: Dict[str, str], attrs: List[str] = ['@cx', '@cy', '@rx', '@ry']) -> None:
        cx, cy, rx, ry = self._extract_attributes(obj, attrs)

        self.cx, self.cy = cx, cy
        self.w, self.h = rx + rx, ry + ry

    def to_yolo_object(self, image_id: str, width: int, height: int, cid: str) -> Dict[str, Union[int, float]]:
        return dict(
            image_id=int(image_id) + 1,
            cid=cid,
            cx=round(self.cx / width, 5),
            cy=round(self.cy / height, 5),
            w=round(self.w / width, 5),
            h=round(self.h / height, 5),
        )
