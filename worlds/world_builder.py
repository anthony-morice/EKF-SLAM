import numpy as np

sphere_def_string = \
'''Solid {{
  translation {0} {1} 0.1
  children [
    Shape {{
      appearance DEF COLOR PBRAppearance {{
        baseColor 1 0.5 0.5
        roughness 1
        metalness 0
      }}
      geometry DEF GEOM Sphere {{
        radius 0.02
      }}
    }}
  ]
  name "Landmark-0"
  recognitionColors [
    1 0.5 0.5
  ]
}}\n'''

sphere_child_string = \
'''Solid {{
  translation {1} {2} 0.13
  children [
    Shape {{
      appearance USE COLOR
      geometry USE GEOM
    }}
  ]
  name "Landmark-{0}"
  recognitionColors [
    1 0.5 0.5
  ]
}}\n'''

def add_spheres_to_world(count, x_range=(-4,4), y_range=(-2,2), filename='project.wbt'):
	coords = [(x_range[1] - x_range[0]) * np.random.random(count) + x_range[0],\
						(y_range[1] - y_range[0]) * np.random.random(count) + y_range[0]]
	spheres = [sphere_def_string.format(coords[0][0], coords[1][0])] + \
						[sphere_child_string.format(i, coords[0][i], coords[1][i]) for i in range(1,count)]
	try:
		f = open(filename, "a")
		f.writelines(spheres)
		f.close()
	except:
		print("Error: could not open world file")

if __name__ == "__main__":
	add_spheres_to_world(200)
