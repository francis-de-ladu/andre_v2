class Coords2:
    def __init__(self, shape: dict[str, (str | int)]) -> None:
        self.frame = shape["frame"] + 1
        self.label = shape["label_id"]

        x1, y1, x2, y2 = shape["points"]

        type_ = shape["type"]
        if type_ == "rectangle":
            self.cx, self.w = self._get_center_and_magnitude(x1, x2)
            self.cy, self.h = self._get_center_and_magnitude(y1, y2)
        elif type_ == "ellipse":
            cx, w = self._get_center_and_magnitude(x1, x2)
            cy, h = self._get_center_and_magnitude(y1, y2)

            self.cx, self.cy = (cx - w / 2), (cy + h / 2)
            self.w, self.h = (w * 2), (h * 2)

    def _get_center_and_magnitude(self, p1: int, p2: int) -> tuple[int, int]:
        center = (p1 + p2) / 2
        magnitude = abs(p2 - p1)
        return center, magnitude

    def to_dict(self) -> dict[str, (str | int)]:
        return dict(
            frame=str(self.frame),
            label=self.label,
            cx=self.cx,
            cy=self.cy,
            w=self.w,
            h=self.h,
        )

    # def to_yolo_object(
    #     self, image_id: str, width: int, height: int, cid: str
    # ) -> Dict[str, Union[int, float]]:
    #     return dict(
    #         image_id=int(image_id) + 1,
    #         cid=cid,
    #         cx=round(self.cx / width, 5),
    #         cy=round(self.cy / height, 5),
    #         w=round(self.w / width, 5),
    #         h=round(self.h / height, 5),
    #     )

    # def to_yolo_crops(
    #     self, image_id: str, block_size: int, cid: str
    # ) -> Dict[str, Union[int, float]]:
    #     return dict(
    #         image_id=str(int(image_id) + 1),
    #         px=int(round(self.cx // block_size, 5)),
    #         py=int(round(self.cy // block_size, 5)),
    #         cid=cid,
    #         cx=round((self.cx % block_size) / block_size, 5),
    #         cy=round((self.cy % block_size) / block_size, 5),
    #         w=round(self.w / block_size, 5),
    #         h=round(self.h / block_size, 5),
    #     )
