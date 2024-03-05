from pathlib import Path
import numpy as np
from gir.dataset.panel_data import PanelData
from gir.config import DATA_PATH

def get_panel_surface(panel: PanelData, res=512):
    """Returns the surface of a panel as a 2D numpy array."""    
    surface_path = Path(DATA_PATH) / f'panel_mat_{res}'
    assert surface_path.is_dir(), f'panel surface path {surface_path} does not exist'
    
    panel_surface_path = surface_path / f'{panel.panel_id}_mat.npz'
    assert panel_surface_path.is_file(), f'panel surface file {panel_surface_path} does not exist'

    panel_surface = np.load(panel_surface_path)
    panel_surface = panel_surface['surface']
    return panel_surface