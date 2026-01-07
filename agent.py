import numpy as np

import tensorflow as tf

from collections import deque



def resize_map_nn(h_map, out_shape=(32, 32)):

    """Nearest-neighbour downsample 2D map (input shape: (D, W)) to target shape."""

    if h_map.size == 0:

        return np.zeros(out_shape, dtype='float32')

    D, W = h_map.shape

    si = np.linspace(0, D - 1, out_shape[0]).astype(int)

    sj = np.linspace(0, W - 1, out_shape[1]).astype(int)

    return h_map[np.ix_(si, sj)]



def prepare_model_input(h_maps, target_shape=(32, 32)):

    """Convert height maps to model input format (n_bins, 32, 32, 1)."""

    arr = np.array(h_maps, dtype='float32')

    if arr.ndim == 2:

        arr = np.expand_dims(arr, 0)

    resized = np.stack([resize_map_nn(h, out_shape=target_shape) for h in arr])

    maxs = np.maximum(1.0, np.max(resized, axis=(1, 2), keepdims=True))

    resized = resized.astype('float32') / maxs

    return resized[..., np.newaxis]



class Agent:

    def __init__(self, env, visualize=False, q_net=None, train=False, verbose=False, batch_size=1):

        self.env = env

        self.visualize = visualize

        self.q_net = q_net

        self.train = train

        self.verbose = verbose

        self.batch_size = batch_size

        self.final_packers = []

        self.eps = 0.0

        self.ep_history = []  # [X] AJOUTER CETTE LIGNE



    def _expected_item_length(self, default=4):

        """Return expected lookahead length (k) for the model's items input.

        Priority:

        1) Infer from `self.q_net` last input tensor shape (batch, k, 3)

        2) Use `env.conveyor.k` if available

        3) Fallback to `default`

        """

        # 1) Inspect the model input shape if possible

        try:

            if self.q_net is not None and hasattr(self.q_net, 'inputs') and self.q_net.inputs:

                shp = self.q_net.inputs[-1].shape  # TensorShape like (None, k, 3)

                if len(shp) >= 3 and shp[1] is not None:

                    return int(shp[1])

        except Exception:

            pass



        # 2) Conveyor-provided k

        try:

            if hasattr(self.env, 'conveyor') and self.env.conveyor is not None and hasattr(self.env.conveyor, 'k'):

                return int(self.env.conveyor.k)

        except Exception:

            pass



        # 3) Fallback

        return default



    # === Nouveau: constructeur de modèle paramétrable par lookahead (k) ===

    def build_q_net(self, lookahead=10, hmap_shape=(32, 32, 1)):

        """Construit un petit réseau multi-entrées:

        - 3 entrées height-map (32x32x1)

        - 1 entrée items (k x 3) normalisée

        Retourne un modèle Keras prêt à l'entraînement si self.train=True.

        """

        import tensorflow as tf



        inputs = []

        features = []



        # 3 branches CNN pour les height maps

        for i in range(3):

            inp = tf.keras.Input(shape=hmap_shape, name=f"hmap_{i+1}")

            x = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inp)

            x = tf.keras.layers.MaxPool2D(pool_size=2)(x)

            x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)

            x = tf.keras.layers.GlobalAveragePooling2D()(x)

            inputs.append(inp)

            features.append(x)



        # Encodage de la séquence d'items (k,3)

        items_in = tf.keras.Input(shape=(lookahead, 3), name='items')

        y = tf.keras.layers.Masking(mask_value=0.0)(items_in)

        y = tf.keras.layers.Flatten()(y)

        y = tf.keras.layers.Dense(64, activation='relu')(y)

        inputs.append(items_in)

        features.append(y)



        # Fusion + tête Q-value

        z = tf.keras.layers.Concatenate()(features)

        z = tf.keras.layers.Dense(128, activation='relu')(z)

        z = tf.keras.layers.Dense(64, activation='relu')(z)

        out = tf.keras.layers.Dense(1, activation='linear', name='q_value')(z)



        model = tf.keras.Model(inputs=inputs, outputs=out, name=f"deeppack_qnet_k{lookahead}")

        if self.train:

            model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

        return model

    

    def _select_action_with_model(self, state, actions):

        """Sélectionne la meilleure action selon le Q-network (multi-input)."""

        items, h_maps, action_list = state

        

        if not self.q_net:

            return self._select_first_valid_action(actions)

        

        # === PRÉPARER LES INPUTS POUR LE MODÈLE ===

        best_action = None

        best_q_value = float('-inf')

        evaluated_actions = 0

        

        try:

            from scipy.ndimage import zoom  # [X] REMPLACER cv2 par scipy

            

            # Préparer les 3 height maps (prendre les 3 premiers bins, padding si nécessaire)

            h_map_inputs = []

            for i in range(3):

                if i < len(h_maps):

                    h_map = h_maps[i]

                else:

                    # Padding avec des zéros si moins de 3 bins

                    h_map = np.zeros((self.env.bin_size[0], self.env.bin_size[1]), dtype='float32')

                

                # Normaliser

                h_map_normalized = h_map / self.env.bin_size[2]

                

                # Redimensionner avec scipy.ndimage.zoom au lieu de cv2.resize

                zoom_factors = (32 / h_map_normalized.shape[0], 32 / h_map_normalized.shape[1])

                h_map_resized = zoom(h_map_normalized, zoom_factors, order=1)  # order=1 = bilinear

                

                h_map_inputs.append(h_map_resized.reshape(1, 32, 32, 1).astype('float32'))

            

            # Préparer les items avec une longueur attendue dynamique (k)

            k_expected = self._expected_item_length(default=4)

            items_input = np.zeros((1, k_expected, 3), dtype='float32')

            limit = min(k_expected, len(items))

            for i in range(limit):

                if items[i] is not None:

                    # Normaliser les dimensions par rapport à la taille du bin

                    items_input[0, i, 0] = items[i][0] / self.env.bin_size[0]

                    items_input[0, i, 1] = items[i][1] / self.env.bin_size[1]

                    items_input[0, i, 2] = items[i][2] / self.env.bin_size[2]

            

            # Combiner tous les inputs

            model_inputs = h_map_inputs + [items_input]

            

            # Prédire la Q-value

            q_values = self.q_net.predict(model_inputs, verbose=0)

            base_q_value = float(q_values[0][0])

            

            if self.verbose:

                print(f"    [...] Q-value de base: {base_q_value:.4f}")

            

            # Évaluer chaque action avec ajustement selon la hauteur

            for item_idx, item_actions in enumerate(action_list):

                for bin_idx, bin_actions in enumerate(item_actions):

                    if not bin_actions:

                        continue

                    

                    # AUGMENTER LE NOMBRE DE POSITIONS ÉVALUÉES

                    n_positions = min(len(bin_actions), 50)  # [X] CHANGÉ de 10 à 30

                    

                    for pos_idx in range(n_positions):

                        position = bin_actions[pos_idx]

                        z_penalty = position[2] / self.env.bin_size[2]

                        adjusted_q = base_q_value - (0.05 * z_penalty)  # [X] 0.05 au lieu de 0.10

                        

                        evaluated_actions += 1

                        

                        if adjusted_q > best_q_value:

                            best_q_value = adjusted_q

                            best_action = (item_idx, bin_idx, pos_idx)

                            

                            if self.verbose:

                                print(f"    [...] Nouvelle meilleure action: item={item_idx}, bin={bin_idx}, pos={pos_idx}, Q={adjusted_q:.4f}")

            

            if self.verbose and best_action:

                print(f"  [X] Action finale: {best_action}, Q={best_q_value:.4f} ({evaluated_actions} évaluées)")

        

        except Exception as e:

            if self.verbose:

                print(f"  [X][X] Erreur dans _select_action_with_model: {e}")

            import traceback

            traceback.print_exc()

            return self._select_first_valid_action(actions)

        

        return best_action if best_action else self._select_first_valid_action(actions)

    

    def run(self, n_episodes=-1, verbose=False):

        """Exécute l'agent pour n_episodes."""

        print(f"DEBUG agent.run(): Starting with n_episodes={n_episodes}")

        

        episode = 0

        while episode != n_episodes:

            print(f"DEBUG agent.run(): Episode {episode} starting...")

            

            state = self.env.reset()

            done = False

            episode_reward = 0

            steps = 0

            utils = []

            

            while not done:

                steps += 1

                if verbose or self.verbose:

                    print(f"  Step {steps}: checking actions...")

                

                items, h_maps, actions = state

                

                if not actions or len(actions) == 0:

                    print(f"    Found 0 valid actions")

                    done = True

                    break

                

                total_actions = sum(

                    sum(len(bin_acts) for bin_acts in item_acts) 

                    for item_acts in actions

                )

                

                if verbose or self.verbose:

                    print(f"    Found {total_actions} valid actions")

                

                # === SÉLECTION D'ACTION AVEC MODÈLE ===

                if self.q_net and np.random.rand() > self.eps:

                    # Exploitation : utiliser le modèle

                    action = self._select_action_with_model(state, actions)

                else:

                    # Exploration ou pas de modèle : greedy

                    action = self._select_first_valid_action(actions)

                

                if action is None:

                    print(f"    No valid action selected, done=True")

                    done = True

                    break

                

                # Exécuter l'action

                next_state, reward, done = self.env.step(action)

                episode_reward += reward

                utils.append(reward)

                state = next_state

            

            print(f"Episode {episode} finished: reward={episode_reward:.4f}, steps={steps}, bins={len(self.env.used_packers)}")

            

            # === ENREGISTRER DANS ep_history ===

            self.ep_history.append((utils, len(self.env.used_packers), episode_reward))

            

            yield (episode, episode_reward, utils)

            

            if n_episodes == 1 or n_episodes == -1:

                print(f"DEBUG agent.run(): Single episode complete, stopping")

                break

            

            episode += 1

        

        print(f"DEBUG agent.run(): All episodes completed")

    

    def _select_first_valid_action(self, actions):

        """Retourne la première action valide."""

        for item_idx, item_actions in enumerate(actions):

            for bin_idx, bin_actions in enumerate(item_actions):

                if bin_actions and len(bin_actions) > 0:

                    return (item_idx, bin_idx, 0)

        return None



class HeuristicAgent:

    def __init__(self, heuristic_fn, env, verbose=False, visualize=False):

        self.heuristic_fn = heuristic_fn

        self.env = env

        self.verbose = verbose

        self.visualize = visualize

        self.ep_history = []

    

    def run(self, n_episodes, verbose=False):

        """Run using a heuristic function."""

        for episode in range(n_episodes):

            state = self.env.reset()

            items, h_maps, actions = state

            

            episode_reward = 0.0

            episode_utils = []

            done = False

            

            while not done:

                if not actions or len(actions) == 0:

                    done = True

                    break

                

                # chercher première action disponible (heuristique simple)

                action = None

                for i, item_actions in enumerate(actions):

                    if item_actions:

                        for j, bin_actions in enumerate(item_actions):

                            if bin_actions:

                                action = (i, j, 0)

                                break

                        if action:

                            break

                

                if action is None:

                    done = True

                else:

                    try:

                        next_state, reward, done = self.env.step(action)

                        items, h_maps, actions = next_state

                        episode_reward += reward

                        episode_utils.append(reward)

                    except Exception as e:

                        if verbose:

                            print(f"Heuristic step error: {e}")

                        done = True

            

            self.ep_history.append((episode_utils, len(self.env.used_packers), episode_reward))

            if verbose:

                print(f"Episode {episode}: reward={episode_reward:.4f}")

            

            yield (episode, episode_reward, episode_utils)