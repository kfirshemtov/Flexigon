import bpy

def reset_shape_keys_to_zero():
    obj = bpy.context.active_object
    if obj and obj.type == 'MESH' and obj.data.shape_keys:
        for key_block in obj.data.shape_keys.key_blocks:
            key_block.value = 0.0
        print(f"All shape keys for '{obj.name}' have been reset to 0.")
    else:
        print("No mesh object with shape keys selected.")

# Run the function
reset_shape_keys_to_zero()
