from .base_options import BaseOptions

class InferenceOptions(BaseOptions):
    def initialize(self):
        
        BaseOptions.initialize(self)
        
        # Required mesh path for inference
        self.parser.add_argument('--mesh_path', type=str, required=True, help='Path to the input mesh file')
        # Model loading parameters
        self.parser.add_argument('--which_epoch', type=str, help='which epoch to load? set to latest to use latest cached model')
        
        # Visualization parameters
        self.parser.add_argument('--show_results', action='store_true', help='Show results in mesh viewer')
        self.parser.add_argument('--save_results', action='store_true', help='Save results to disk')
        self.parser.add_argument('--results_dir', type=str, help='Directory to save results')
        
        # Set to test mode
        self.is_train = False 
        
        
        