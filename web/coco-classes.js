// Nombres de clases y orden usados por best.pt.
const COCO_CLASSES = [
  "bicycle", "bridge", "building", "bus", "car", "caravan", "fence",
  "guard rail", "motorcycle", "parking", "person", "pole", "rail track",
  "rider", "road", "sidewalk", "sky", "terrain", "traffic light",
  "traffic sign", "trailer", "train", "trash container", "truck", "tunnel",
  "vegetation", "wall"
];

// Genera un color estable (HSL -> "rgb(r,g,b)") por índice de clase
function classColor(classId) {
  const hue = (classId * 37) % 360;
  return `hsl(${hue}, 85%, 55%)`;
}
