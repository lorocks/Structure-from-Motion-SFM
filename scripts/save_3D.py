import numpy as np

def save_ply(point_clouds, colors, output_dir):
    out_points = point_clouds.reshape(-1, 3) * 200
    out_colors = colors.reshape(-1, 3)
    # out_colors = np.zeros(out_points.shape)
    print(f"out_colors shape: {out_colors.shape}, out_points shape: {out_points.shape}")
    verts = np.hstack([out_points, out_colors])

    mean = np.mean(verts[:, :3], axis=0)
    scaled_verts = verts[:, :3] - mean
    dist = np.sqrt(scaled_verts[:, 0] ** 2 + scaled_verts[:, 1] ** 2 + scaled_verts[:, 2] ** 2)
    indx = np.where(dist < np.mean(dist) + 300)

    verts = verts[indx]
    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar blue
        property uchar green
        property uchar red
        end_header
        '''
    
    with open(output_dir, 'w') as f:
        f.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(f, verts, '%f %f %f %d %d %d')