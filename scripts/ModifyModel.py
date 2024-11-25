import bpy

blend_file_path = "3dmodel.blend"

bpy.ops.wm.open_mainfile(filepath=blend_file_path)

def scale_chair_and_table(scale_x, scale_z_chair, scale_z_table):
    # Locate the chair object by name
    chair_obj = bpy.data.objects.get("chair")
    if chair_obj:
        chair_obj.scale[0] = scale_x  
        chair_obj.scale[2] = scale_z_chair  
        print(f"Scaled 'chair' object to X: {scale_x}, Z: {scale_z_chair}")
    else:
        print("Error: No object named 'chair' found in the scene")
    
    # Locate the table_top object by name
    table_top_obj = bpy.data.objects.get("table_top")
    if table_top_obj:
        table_top_obj.scale[2] = scale_z_table
        print(f"Scaled 'table_top' object to Z: {scale_z_table}")
    else:
        print("Error: No object named 'table_top' found in the scene")

# Define the scale values
scale_x_value = 0.03 # chair width 
scale_z_chair_value = 0.02 #chair height
scale_z_table_value = 1.5  # Adjust this to the desired Z scale for table_top

# Call the function with the specified values
scale_chair_and_table(scale_x_value, scale_z_chair_value, scale_z_table_value)

# Save the modified file
output_glb_path = "Scaled3dModel.glb"
bpy.ops.export_scene.gltf(filepath=output_glb_path, export_format='GLB')

