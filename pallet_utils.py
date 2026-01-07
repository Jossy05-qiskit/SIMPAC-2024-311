"""
Utilitaires pour gérer la palette et les coordonnées absolues.
"""

class PalletConfig:
    """Configuration de la palette industrielle."""
    def __init__(self, 
                 pallet_x=1295.889166,      # Position palette (centro)
                 pallet_y=9096.996345,      # Position palette (centro)
                 pallet_z=0,                # Sol (z=0)
                 pallet_length=1200,        # mm
                 pallet_width=1200,         # mm
                 pallet_height=145,         # mm (à ajouter à z des cartons)
                 bin_size=(1200, 1200, 2000)):  # Taille du bac virtuel
        self.pallet_x = pallet_x
        self.pallet_y = pallet_y
        self.pallet_z = pallet_z
        self.pallet_length = pallet_length
        self.pallet_width = pallet_width
        self.pallet_height = pallet_height
        self.bin_size = bin_size
        
    def local_to_absolute(self, local_x, local_y, local_z):
        """
        Convertit les coordonnées locales (depuis DeepPack3D) 
        en coordonnées absolues (système DELMIA).
        
        local_x, local_y : dans le repère palette (coin inférieur gauche = 0,0)
        local_z : hauteur relative au sol
        """
        # Position absolue = position palette + offset local
        abs_x = self.pallet_x - self.pallet_length / 2.0 + local_x
        abs_y = self.pallet_y - self.pallet_width / 2.0 + local_y
        abs_z = self.pallet_z + self.pallet_height + local_z  # +height palette
        
        return abs_x, abs_y, abs_z
    
    def absolute_to_local(self, abs_x, abs_y, abs_z):
        """Inverse : coordonnées absolues → locales (pour debug)."""
        local_x = abs_x - (self.pallet_x - self.pallet_length / 2.0)
        local_y = abs_y - (self.pallet_y - self.pallet_width / 2.0)
        local_z = abs_z - self.pallet_z - self.pallet_height
        return local_x, local_y, local_z


def scale_item(item_size, input_scale=1.0, output_scale=1.0):
    """
    Redimensionne un item pour adapter input_pack (mm) 
    vers le système d'unités interne.
    
    Example:
        item = (500, 500, 500)  # mm
        scaled = scale_item(item, input_scale=1.0, output_scale=1.0)  # pas de changement
    """
    w, h, d = item_size
    return (int(w * output_scale / input_scale), 
            int(h * output_scale / input_scale),
            int(d * output_scale / input_scale))