from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch


def _read_csv(path: Path) -> torch.Tensor:
    return torch.tensor(pd.read_csv(path, header=None).values, dtype=torch.float32)


@dataclass
class StateCouplingTopology:
    H: torch.Tensor
    noise_std: torch.Tensor
    rinv: torch.Tensor
    g_inv: torch.Tensor
    weighted_H: torch.Tensor
    weighted_leverage: torch.Tensor
    measurement_type_ids: torch.Tensor
    measurement_bus_ids: torch.Tensor
    measurement_from_bus_ids: torch.Tensor
    measurement_to_bus_ids: torch.Tensor
    measurement_branch_ids: torch.Tensor
    measurement_adjacency: torch.Tensor
    measurement_effectiveness: torch.Tensor
    measurement_stealth: torch.Tensor
    measurement_topology_score: torch.Tensor
    measurement_state_smoothness: torch.Tensor
    measurement_projection_locality: torch.Tensor
    explicit_adjacency: torch.Tensor | None
    adjacency: torch.Tensor
    # Weighted unit state-to-measurement gain: sqrt(diag(H^T R^-1 H)).
    # This is a structural gain proxy, not a sample-specific vulnerability score.
    state_importance: torch.Tensor
    state_measurement_support_count: torch.Tensor
    state_injection_gain: torch.Tensor
    state_flow_gain: torch.Tensor
    state_topology_centrality: torch.Tensor
    state_local_redundancy: torch.Tensor
    state_flow_support: torch.Tensor
    state_corridor_centrality: torch.Tensor
    state_degree: torch.Tensor
    state_bus_ids: torch.Tensor
    graph_source: str
    noise_model: str

    @staticmethod
    def _row_normalize(adjacency: torch.Tensor) -> torch.Tensor:
        adjacency = adjacency.clone()
        adjacency.fill_diagonal_(0.0)
        row_sum = adjacency.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return adjacency / row_sum

    @staticmethod
    def _normalize_vector(values: torch.Tensor) -> torch.Tensor:
        values = values.reshape(-1).to(dtype=torch.float32)
        vmax = float(values.max().item()) if values.numel() > 0 else 0.0
        if vmax <= 0.0:
            return torch.zeros_like(values)
        return values / vmax

    @staticmethod
    def _graph_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
        adjacency = adjacency.to(dtype=torch.float32)
        sym_adj = 0.5 * (adjacency + adjacency.T)
        sym_adj = sym_adj.clone()
        sym_adj.fill_diagonal_(0.0)
        degree = sym_adj.sum(dim=1)
        return torch.diag(degree) - sym_adj

    @staticmethod
    def _graph_smoothness_score(signals: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        signals = signals.to(dtype=torch.float32)
        if signals.ndim == 1:
            signals = signals.unsqueeze(0)
        laplacian = laplacian.to(dtype=signals.dtype, device=signals.device)
        variation = ((signals @ laplacian) * signals).sum(dim=1).clamp(min=0.0)
        norm_sq = signals.pow(2).sum(dim=1).clamp(min=1e-8)
        return 1.0 / (1.0 + variation / norm_sq)

    @staticmethod
    def _build_explicit_topology(
        base: Path,
        n_states: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, str]:
        state_graph_path = base / "state_graph_edges.csv"
        state_bus_map_path = base / "state_bus_map.csv"
        if not (state_graph_path.exists() and state_bus_map_path.exists()):
            return (
                None,
                torch.arange(1, n_states + 1, dtype=torch.float32),
                torch.zeros(n_states, dtype=torch.float32),
                "jacobian_only",
            )

        edges = _read_csv(state_graph_path)
        state_bus_map = _read_csv(state_bus_map_path)
        explicit_adj = torch.zeros(n_states, n_states, dtype=torch.float32)

        if edges.ndim == 1:
            edges = edges.reshape(1, -1)
        for row in edges:
            src = int(row[0].item()) - 1
            dst = int(row[1].item()) - 1
            if src < 0 or dst < 0 or src >= n_states or dst >= n_states:
                continue
            weight = float(row[3].item()) if row.numel() >= 4 else 1.0
            explicit_adj[src, dst] += weight
            explicit_adj[dst, src] += weight

        if state_bus_map.ndim == 1:
            state_bus_map = state_bus_map.reshape(1, -1)
        state_rows = state_bus_map[state_bus_map[:, 3] > 0]
        state_rows = state_rows[torch.argsort(state_rows[:, 3])]

        state_bus_ids = state_rows[:, 1].clone() if state_rows.shape[0] == n_states else torch.arange(
            1, n_states + 1, dtype=torch.float32
        )
        state_degree = state_rows[:, 4].clone() if state_rows.shape[0] == n_states else torch.zeros(
            n_states, dtype=torch.float32
        )
        return explicit_adj, state_bus_ids, state_degree, "explicit_bus_branch"

    @staticmethod
    def _default_measurement_types(
        n_measurements: int,
        n_states: int,
    ) -> torch.Tensor:
        n_bus = n_states + 1
        n_branch = max(0, (n_measurements - n_bus) // 2)
        types = torch.ones(n_measurements, dtype=torch.long)
        pf_start = min(n_bus, n_measurements)
        pf_end = min(pf_start + n_branch, n_measurements)
        pt_start = pf_end
        types[pf_start:pf_end] = 2
        types[pt_start:] = 3
        return types

    @classmethod
    def _load_measurement_metadata(
        cls,
        base: Path,
        n_measurements: int,
        n_states: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        measurement_map_path = base / "measurement_map.csv"
        default_types = cls._default_measurement_types(
            n_measurements=n_measurements,
            n_states=n_states,
        )
        default_bus = torch.zeros(n_measurements, dtype=torch.float32)
        default_from = torch.zeros(n_measurements, dtype=torch.float32)
        default_to = torch.zeros(n_measurements, dtype=torch.float32)
        default_branch = torch.zeros(n_measurements, dtype=torch.float32)
        if not measurement_map_path.exists():
            return (
                default_types,
                default_bus,
                default_from,
                default_to,
                default_branch,
            )

        measurement_map = _read_csv(measurement_map_path)
        if measurement_map.ndim == 1:
            measurement_map = measurement_map.reshape(1, -1)
        if measurement_map.shape[0] != n_measurements or measurement_map.shape[1] < 2:
            return (
                default_types,
                default_bus,
                default_from,
                default_to,
                default_branch,
            )

        types = torch.round(measurement_map[:, 1]).to(dtype=torch.long)
        valid = (types >= 1) & (types <= 3)
        if not bool(valid.all().item()):
            types = default_types

        if measurement_map.shape[1] >= 6:
            bus_ids = measurement_map[:, 2].clone()
            from_bus_ids = measurement_map[:, 3].clone()
            to_bus_ids = measurement_map[:, 4].clone()
            branch_ids = measurement_map[:, 5].clone()
        else:
            bus_ids = default_bus
            from_bus_ids = default_from
            to_bus_ids = default_to
            branch_ids = default_branch
        return (
            types,
            bus_ids.to(dtype=torch.float32),
            from_bus_ids.to(dtype=torch.float32),
            to_bus_ids.to(dtype=torch.float32),
            branch_ids.to(dtype=torch.float32),
        )

    @classmethod
    def from_data_dir(
        cls,
        data_dir: str,
        topology_mode: str = "auto",
        noise_model: str = "unknown",
    ) -> "StateCouplingTopology":
        base = Path(data_dir)
        H = _read_csv(base / "jacobian_H_adjust.csv")
        requested_noise_model = str(noise_model).lower()
        if requested_noise_model not in {"known", "unknown", "isotropic"}:
            raise ValueError(
                "Unsupported noise_model. Expected one of: known, unknown, isotropic."
            )
        # Do not use measurement-noise covariance as prior knowledge in attack guidance.
        # Keep identity weighting so physical priors only rely on topology and Jacobian H.
        noise_model = "unknown"
        noise_std = torch.ones(H.shape[0], dtype=torch.float32)
        rinv = 1.0 / (noise_std.pow(2))
        weighted_H = H / noise_std.unsqueeze(1)
        weighted_normal = H.T @ (H * rinv.unsqueeze(1))
        weighted_normal_pinv = torch.linalg.pinv(weighted_normal)
        weighted_leverage = (
            (weighted_H @ weighted_normal_pinv) * weighted_H
        ).sum(dim=1).clamp(min=0.0)
        (
            measurement_type_ids,
            measurement_bus_ids,
            measurement_from_bus_ids,
            measurement_to_bus_ids,
            measurement_branch_ids,
        ) = cls._load_measurement_metadata(
            base=base,
            n_measurements=H.shape[0],
            n_states=H.shape[1],
        )
        flow_mask = measurement_type_ids != 1

        support = (weighted_H.abs() > 1e-8).float()
        overlap = support.T @ support
        corr = weighted_H.T.abs() @ weighted_H.abs()
        corr.fill_diagonal_(0.0)
        corr = cls._row_normalize(corr / corr.max().clamp(min=1e-8))

        if topology_mode not in {"auto", "explicit", "jacobian"}:
            raise ValueError(
                "Unsupported topology_mode. Expected one of: auto, explicit, jacobian."
            )

        explicit_adj = None
        state_bus_ids = torch.arange(1, H.shape[1] + 1, dtype=torch.float32)
        state_degree = torch.zeros(H.shape[1], dtype=torch.float32)
        graph_source = "jacobian_only"
        if topology_mode != "jacobian":
            explicit_adj, state_bus_ids, state_degree, graph_source = cls._build_explicit_topology(
                base=base,
                n_states=H.shape[1],
            )
            if topology_mode == "explicit" and explicit_adj is None:
                raise FileNotFoundError(
                    f"Explicit topology metadata not found in {base}."
                )

        if explicit_adj is not None:
            topo_adj = cls._row_normalize(explicit_adj)
            topo_two_hop = cls._row_normalize(topo_adj @ topo_adj)
            adjacency = topo_adj + 0.35 * topo_two_hop + 0.15 * corr
            state_corridor_centrality = cls._normalize_vector(explicit_adj.sum(dim=1))
            topo_centrality_base = 0.5 * (explicit_adj + explicit_adj.T)
        else:
            adjacency = overlap + corr
            state_corridor_centrality = cls._normalize_vector(state_degree)
            topo_centrality_base = 0.5 * (adjacency + adjacency.T)
        adjacency = cls._row_normalize(adjacency)
        # Equivalent to ||R^{-1/2} h_i||_2 for each Jacobian column h_i, i.e.
        # sqrt(diag(H^T R^{-1} H)). Using the diagonal form makes the Fisher-style
        # interpretation explicit while preserving the same ranking as the column norm.
        state_importance = torch.sqrt(torch.diagonal(weighted_normal).clamp(min=0.0))
        weighted_sq = H.pow(2) * rinv.unsqueeze(1)
        if bool((measurement_type_ids == 1).any().item()):
            state_injection_gain = torch.sqrt(
                weighted_sq[measurement_type_ids == 1].sum(dim=0).clamp(min=0.0)
            )
        else:
            state_injection_gain = torch.zeros(H.shape[1], dtype=torch.float32)
        if bool((measurement_type_ids != 1).any().item()):
            state_flow_gain = torch.sqrt(
                weighted_sq[measurement_type_ids != 1].sum(dim=0).clamp(min=0.0)
            )
        else:
            state_flow_gain = torch.zeros(H.shape[1], dtype=torch.float32)
        total_state_mass = H.abs().sum(dim=0).clamp(min=1e-8)
        if bool(flow_mask.any().item()):
            state_flow_support = H[flow_mask].abs().sum(dim=0) / total_state_mass
        else:
            state_flow_support = torch.zeros(H.shape[1], dtype=torch.float32)

        estimator = weighted_normal_pinv @ (H.T * rinv.unsqueeze(0))
        measurement_effectiveness = torch.linalg.norm(estimator, dim=0)
        residual_sensitivity = (
            torch.sqrt((1.0 - weighted_leverage).clamp(min=0.0)) / noise_std.clamp(min=1e-8)
        )
        measurement_stealth = 1.0 - cls._normalize_vector(residual_sensitivity)
        state_physical_score = cls._normalize_vector(
            0.45 * cls._normalize_vector(state_importance)
            + 0.30 * cls._normalize_vector(state_flow_support)
            + 0.25 * cls._normalize_vector(state_corridor_centrality)
        )
        row_mass = H.abs().sum(dim=1).clamp(min=1e-8)
        measurement_topology_score = (H.abs() @ state_physical_score) / row_mass

        measurement_overlap = support @ support.T
        measurement_overlap.fill_diagonal_(0.0)
        measurement_overlap = cls._row_normalize(measurement_overlap)

        max_bus_id = int(
            max(
                float(measurement_bus_ids.max().item()),
                float(measurement_from_bus_ids.max().item()),
                float(measurement_to_bus_ids.max().item()),
                0.0,
            )
        )
        if max_bus_id > 0:
            bus_assoc = torch.zeros(H.shape[0], max_bus_id, dtype=torch.float32)
            for row_idx in range(H.shape[0]):
                assoc_ids = set()
                for value in (
                    measurement_bus_ids[row_idx],
                    measurement_from_bus_ids[row_idx],
                    measurement_to_bus_ids[row_idx],
                ):
                    bus_id = int(round(float(value.item())))
                    if bus_id > 0:
                        assoc_ids.add(bus_id)
                for bus_id in assoc_ids:
                    bus_assoc[row_idx, bus_id - 1] = 1.0
            bus_share = bus_assoc @ bus_assoc.T
            bus_share.fill_diagonal_(0.0)
            bus_share = cls._row_normalize(bus_share)
        else:
            bus_share = torch.zeros_like(measurement_overlap)

        valid_branch = measurement_branch_ids > 0
        branch_share = (
            (measurement_branch_ids.reshape(-1, 1) == measurement_branch_ids.reshape(1, -1))
            & valid_branch.reshape(-1, 1)
            & valid_branch.reshape(1, -1)
        ).to(dtype=torch.float32)
        branch_share.fill_diagonal_(0.0)
        branch_share = cls._row_normalize(branch_share)

        measurement_adjacency = cls._row_normalize(
            measurement_overlap + 0.35 * bus_share + 0.20 * branch_share
        )
        topo_centrality = torch.zeros(H.shape[1], dtype=torch.float32)
        if float(topo_centrality_base.abs().sum().item()) > 0.0:
            topo_centrality_base = topo_centrality_base.to(dtype=torch.float32)
            topo_centrality_base = topo_centrality_base.clone()
            topo_centrality_base.fill_diagonal_(0.0)
            try:
                eigvals, eigvecs = torch.linalg.eigh(topo_centrality_base)
                topo_centrality = cls._normalize_vector(torch.abs(eigvecs[:, -1]))
            except RuntimeError:
                topo_centrality = cls._normalize_vector(topo_centrality_base.sum(dim=1))
        measure_count = support.sum(dim=0)
        measure_count_norm = cls._normalize_vector(measure_count)
        leverage_norm = cls._normalize_vector(weighted_leverage)
        local_density = torch.zeros(H.shape[1], dtype=torch.float32)
        local_leverage = torch.zeros(H.shape[1], dtype=torch.float32)
        for state_idx in range(H.shape[1]):
            assoc = support[:, state_idx].to(dtype=torch.float32)
            assoc_count = float(assoc.sum().item())
            if assoc_count <= 0.0:
                continue
            local_density[state_idx] = float(
                (assoc @ measurement_adjacency @ assoc).item() / max(assoc_count * assoc_count, 1e-8)
            )
            local_leverage[state_idx] = float(
                (assoc * leverage_norm).sum().item() / max(assoc_count, 1e-8)
            )
        state_local_redundancy = cls._normalize_vector(
            0.40 * measure_count_norm
            + 0.35 * cls._normalize_vector(local_density)
            + 0.25 * cls._normalize_vector(local_leverage)
        )
        state_laplacian = cls._graph_laplacian(adjacency)
        measurement_laplacian = cls._graph_laplacian(measurement_adjacency)
        state_response = estimator.T
        projected_response = (H @ estimator).T
        measurement_state_smoothness = cls._normalize_vector(
            cls._graph_smoothness_score(state_response, state_laplacian)
        )
        measurement_projection_locality = cls._normalize_vector(
            cls._graph_smoothness_score(projected_response, measurement_laplacian)
        )
        return cls(
            H=H,
            noise_std=noise_std,
            rinv=rinv,
            g_inv=weighted_normal_pinv,
            weighted_H=weighted_H,
            weighted_leverage=weighted_leverage,
            measurement_type_ids=measurement_type_ids,
            measurement_bus_ids=measurement_bus_ids,
            measurement_from_bus_ids=measurement_from_bus_ids,
            measurement_to_bus_ids=measurement_to_bus_ids,
            measurement_branch_ids=measurement_branch_ids,
            measurement_adjacency=measurement_adjacency,
            measurement_effectiveness=measurement_effectiveness.to(dtype=torch.float32),
            measurement_stealth=measurement_stealth.to(dtype=torch.float32),
            measurement_topology_score=measurement_topology_score.to(dtype=torch.float32),
            measurement_state_smoothness=measurement_state_smoothness.to(dtype=torch.float32),
            measurement_projection_locality=measurement_projection_locality.to(dtype=torch.float32),
            explicit_adjacency=explicit_adj,
            adjacency=adjacency,
            state_importance=state_importance,
            state_measurement_support_count=measure_count.to(dtype=torch.float32),
            state_injection_gain=state_injection_gain.to(dtype=torch.float32),
            state_flow_gain=state_flow_gain.to(dtype=torch.float32),
            state_topology_centrality=topo_centrality.to(dtype=torch.float32),
            state_local_redundancy=state_local_redundancy.to(dtype=torch.float32),
            state_flow_support=state_flow_support.to(dtype=torch.float32),
            state_corridor_centrality=state_corridor_centrality.to(dtype=torch.float32),
            state_degree=state_degree,
            state_bus_ids=state_bus_ids,
            graph_source=graph_source,
            noise_model=str(noise_model),
        )

    @property
    def n_states(self) -> int:
        return int(self.H.shape[1])

    @property
    def n_measurements(self) -> int:
        return int(self.H.shape[0])

    def project_measurement_to_col_space(self, delta_z: torch.Tensor) -> torch.Tensor:
        delta_z = delta_z.to(dtype=torch.float32)
        if delta_z.ndim == 1:
            delta_z = delta_z.unsqueeze(0)
        coeff = (delta_z * self.rinv.reshape(1, -1)) @ self.H @ self.g_inv
        return coeff @ self.H.T

    def estimate_state_from_measurement(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(dtype=torch.float32)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        return (z * self.rinv.reshape(1, -1)) @ self.H @ self.g_inv

    def project_measurement_reference(self, z: torch.Tensor) -> torch.Tensor:
        theta_hat = self.estimate_state_from_measurement(z)
        return theta_hat @ self.H.T

    def project_measurement_to_state_region(
        self,
        delta_z: torch.Tensor,
        state_region: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta_z = delta_z.to(dtype=torch.float32)
        if delta_z.ndim == 1:
            delta_z = delta_z.unsqueeze(0)
        state_region = state_region.reshape(-1).long()
        if state_region.numel() == 0:
            empty = torch.zeros(delta_z.shape[0], 0, dtype=torch.float32, device=delta_z.device)
            return empty, torch.zeros_like(delta_z)

        region_h = self.H[:, state_region].to(device=delta_z.device, dtype=delta_z.dtype)
        region_rinv = self.rinv.to(device=delta_z.device, dtype=delta_z.dtype)
        gram = region_h.T @ (region_h * region_rinv.unsqueeze(1))
        gram_inv = torch.linalg.pinv(gram)
        coeff = (delta_z * region_rinv.unsqueeze(0)) @ region_h @ gram_inv
        projected = coeff @ region_h.T
        return coeff, projected

    def measurement_region_to_state_region(
        self,
        region: torch.Tensor,
        x_ref: torch.Tensor,
        region_size: int,
        delta_state_hint: torch.Tensor | None = None,
    ) -> torch.Tensor:
        region = region.reshape(-1).long()
        region_size = max(1, min(int(region_size), self.n_states))
        if region.numel() == 0:
            return torch.arange(region_size, dtype=torch.long)

        state_score, _ = self.state_guidance_from_measurement(x_ref)
        local_h = self.H[region].to(dtype=torch.float32)
        local_rinv = self.rinv[region].to(dtype=torch.float32)
        backproj = self.g_inv.to(dtype=torch.float32) @ (
            local_h.T * local_rinv.reshape(1, -1)
        )
        influence = self._normalize_vector(torch.linalg.norm(backproj, dim=1))
        hint = torch.zeros_like(influence)
        if delta_state_hint is not None:
            hint = self._normalize_vector(delta_state_hint.reshape(-1).abs())
        score = self._normalize_vector(
            0.50 * influence
            + 0.30 * state_score
            + 0.20 * hint
        )
        diffused = self._normalize_vector(0.75 * score + 0.25 * (self.adjacency @ score))
        return torch.topk(diffused, k=region_size).indices.to(dtype=torch.long)

    def _legacy_seed_score(self, c_base: torch.Tensor) -> torch.Tensor:
        c_base = c_base.reshape(-1).to(dtype=torch.float32)
        seed_score = c_base.abs() * self.state_importance
        if float(seed_score.max().item()) <= 0.0:
            seed_score = self.state_importance.clone()
        if self.state_degree.numel() == seed_score.numel() and float(
            self.state_degree.max().item()
        ) > 0.0:
            degree_gain = self.state_degree / self.state_degree.max().clamp(min=1e-8)
            seed_score = seed_score * (1.0 + 0.10 * degree_gain)
        return self._normalize_vector(seed_score)

    def _state_coupling_weights(self) -> torch.Tensor:
        if self.explicit_adjacency is not None:
            coupling = self.explicit_adjacency.to(dtype=torch.float32).clone()
        else:
            coupling = self.adjacency.to(dtype=torch.float32).clone()
        coupling = 0.5 * (coupling + coupling.T)
        coupling.fill_diagonal_(0.0)
        return coupling

    def state_target_radiated_prior(self, c_base: torch.Tensor) -> torch.Tensor:
        c_base = c_base.reshape(-1).to(dtype=torch.float32)
        if int(c_base.numel()) != int(self.n_states):
            raise ValueError(
                f"State seed dimension mismatch: got {int(c_base.numel())}, expected {int(self.n_states)}."
            )

        legacy_seed = self._legacy_seed_score(c_base)
        leverage = self._normalize_vector(self.state_importance)
        target_mask = c_base.abs() > 1e-10
        non_target_mask = ~target_mask
        if not bool(non_target_mask.any().item()) or not bool(target_mask.any().item()):
            return legacy_seed

        target_mass = torch.where(target_mask, c_base.abs(), torch.zeros_like(c_base))
        radiated = self._state_coupling_weights().T @ target_mass
        radiated = self._normalize_vector(radiated)
        radiated_non_target = torch.where(
            non_target_mask,
            radiated * leverage,
            torch.zeros_like(leverage),
        )
        score = torch.where(target_mask, legacy_seed, radiated_non_target)

        if float(score.max().item()) <= 0.0:
            return legacy_seed
        return self._normalize_vector(score)

    def _seed_score(self, c_base: torch.Tensor) -> torch.Tensor:
        seed_score = self._legacy_seed_score(c_base)
        if float(seed_score.max().item()) <= 0.0:
            return self._normalize_vector(self.state_importance)
        return seed_score

    def _diffuse_seed_score(
        self,
        seed_score: torch.Tensor,
        diffusion_steps: int,
        diffusion_alpha: float,
    ) -> torch.Tensor:
        steps = max(0, int(diffusion_steps))
        alpha = float(min(max(diffusion_alpha, 0.0), 1.0))
        score = self._normalize_vector(seed_score)
        if steps <= 0 or alpha <= 0.0:
            return score

        sym_adj = 0.5 * (self.adjacency + self.adjacency.T)
        for _ in range(steps):
            score = (1.0 - alpha) * score + alpha * (sym_adj @ score)
            score = self._normalize_vector(score)
        return score

    @staticmethod
    def _sample_index_pool(
        score: torch.Tensor,
        pool_size: int,
        generator: torch.Generator | None = None,
        temperature: float = 1.0,
        uniform_mixing: float = 0.0,
    ) -> list[int]:
        score = score.reshape(-1).to(dtype=torch.float32)
        if score.numel() <= 0:
            return []
        pool_size = max(1, min(int(pool_size), int(score.numel())))
        temperature = max(float(temperature), 1e-4)
        uniform_mixing = float(min(max(uniform_mixing, 0.0), 1.0))

        logits = score / temperature
        logits = logits - logits.max()
        probs = torch.softmax(logits, dim=0)
        if uniform_mixing > 0.0:
            uniform = torch.full_like(probs, 1.0 / max(1, int(probs.numel())))
            probs = (1.0 - uniform_mixing) * probs + uniform_mixing * uniform
        probs = probs / probs.sum().clamp(min=1e-8)

        try:
            sampled = torch.multinomial(
                probs,
                num_samples=pool_size,
                replacement=False,
                generator=generator,
            )
            return [int(idx) for idx in sampled.tolist()]
        except RuntimeError:
            return torch.topk(score, k=pool_size).indices.tolist()

    def _corridor_seed_score(
        self,
        seed_score: torch.Tensor,
        diffusion_steps: int,
        diffusion_alpha: float,
        flow_weight: float,
        corridor_weight: float,
    ) -> torch.Tensor:
        diffused = self._diffuse_seed_score(
            seed_score=seed_score,
            diffusion_steps=diffusion_steps,
            diffusion_alpha=diffusion_alpha,
        )
        flow_support = self._normalize_vector(self.state_flow_support)
        corridor_support = self._normalize_vector(self.state_corridor_centrality)
        corridor_score = (
            diffused
            + float(flow_weight) * flow_support
            + float(corridor_weight) * corridor_support
        )
        return self._normalize_vector(corridor_score)

    def _strongest_path(self, src: int, dst: int) -> list[int]:
        if (
            self.explicit_adjacency is None
            or src < 0
            or dst < 0
            or src >= self.n_states
            or dst >= self.n_states
        ):
            return [int(src)] if src >= 0 else []
        if src == dst:
            return [int(src)]

        weights = self.explicit_adjacency
        inf = float("inf")
        dist = [inf] * self.n_states
        visited = [False] * self.n_states
        prev = [-1] * self.n_states
        dist[int(src)] = 0.0

        for _ in range(self.n_states):
            best_idx = -1
            best_dist = inf
            for idx in range(self.n_states):
                if visited[idx]:
                    continue
                if dist[idx] < best_dist:
                    best_dist = dist[idx]
                    best_idx = idx
            if best_idx < 0 or best_idx == int(dst):
                break

            visited[best_idx] = True
            neighbors = torch.nonzero(weights[best_idx] > 0, as_tuple=False).reshape(-1)
            for nbr_tensor in neighbors:
                nbr = int(nbr_tensor.item())
                edge_weight = float(weights[best_idx, nbr].item())
                if edge_weight <= 0.0:
                    continue
                edge_cost = 1.0 / max(edge_weight, 1e-8)
                cand = dist[best_idx] + edge_cost
                if cand < dist[nbr]:
                    dist[nbr] = cand
                    prev[nbr] = best_idx

        if dist[int(dst)] >= inf:
            return [int(src), int(dst)]

        path = [int(dst)]
        cur = int(dst)
        while cur != int(src) and prev[cur] >= 0:
            cur = prev[cur]
            path.append(cur)
        path.reverse()
        return path

    def _expand_region(
        self,
        anchors: list[int],
        priority_score: torch.Tensor,
        region_size: int,
        adjacency_matrix: torch.Tensor | None = None,
        importance_values: torch.Tensor | None = None,
        neighbor_weight: float = 1.0,
        priority_weight: float = 0.25,
        importance_weight: float = 0.10,
        allowed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if allowed_mask is not None:
            allowed_mask = allowed_mask.reshape(-1).to(dtype=torch.bool)
        selected = [
            int(i)
            for i in dict.fromkeys(int(i) for i in anchors)
            if allowed_mask is None or bool(allowed_mask[int(i)].item())
        ]
        priority_score = priority_score.reshape(-1).to(dtype=torch.float32)
        if adjacency_matrix is None:
            adjacency_matrix = self.adjacency
        if importance_values is None:
            importance_values = self.state_importance
        importance_score = self._normalize_vector(importance_values)
        if not selected:
            ranked = torch.argsort(priority_score, descending=True).tolist()
            for idx in ranked:
                idx = int(idx)
                if allowed_mask is None or bool(allowed_mask[idx].item()):
                    selected = [idx]
                    break
        if not selected:
            selected = [0]
        while len(selected) < region_size:
            neighbor_score = adjacency_matrix[selected].sum(dim=0)
            combined = (
                float(neighbor_weight) * neighbor_score
                + float(priority_weight) * priority_score
                + float(importance_weight) * importance_score
            )
            if allowed_mask is not None:
                combined = combined.masked_fill(~allowed_mask, -1.0)
            for idx in selected:
                combined[idx] = -1.0
            next_idx = int(torch.argmax(combined).item())
            if next_idx in selected:
                remaining = [
                    idx
                    for idx in torch.argsort(priority_score, descending=True).tolist()
                    if idx not in selected
                    and (allowed_mask is None or bool(allowed_mask[int(idx)].item()))
                ]
                if not remaining:
                    break
                next_idx = int(remaining[0])
            selected.append(next_idx)
        return torch.tensor(selected, dtype=torch.long)

    def state_guidance_from_measurement(
        self,
        x_ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_ref = x_ref.reshape(-1).to(dtype=torch.float32)
        theta_hat = self.estimate_state_from_measurement(x_ref).reshape(-1)
        state_activity = self._normalize_vector(theta_hat.abs())
        state_structure = self._normalize_vector(
            0.55 * self.state_physical_score()
            + 0.45 * self._normalize_vector(self.state_importance)
        )
        state_score = self._normalize_vector(0.65 * state_activity + 0.35 * state_structure)
        sign = torch.sign(theta_hat)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return state_score, state_score * sign

    def state_physical_score(self) -> torch.Tensor:
        return self._normalize_vector(
            0.45 * self._normalize_vector(self.state_importance)
            + 0.30 * self._normalize_vector(self.state_flow_support)
            + 0.25 * self._normalize_vector(self.state_corridor_centrality)
        )

    def state_multisource_prior(self, c_base: torch.Tensor) -> torch.Tensor:
        activity = self.state_activity_score(c_base)
        alignment = self.state_attack_alignment(c_base)
        base_signal = self._normalize_vector(0.70 * activity + 0.30 * alignment)
        hetero_gain = self._normalize_vector(
            0.35 * self._normalize_vector(self.state_injection_gain)
            + 0.65 * self._normalize_vector(self.state_flow_gain)
        )
        topology_gain = self._normalize_vector(
            0.60 * self._normalize_vector(self.state_topology_centrality)
            + 0.40 * self._normalize_vector(self.state_corridor_centrality)
        )
        redundancy_gate = 1.0 / (1.0 + self.state_local_redundancy.to(dtype=torch.float32))
        redundancy_gate = self._normalize_vector(redundancy_gate)
        fused = base_signal * hetero_gain * (1.0 + 0.50 * topology_gain) * redundancy_gate
        if float(fused.max().item()) <= 0.0:
            fused = hetero_gain * (1.0 + 0.50 * topology_gain) * redundancy_gate
        return self._normalize_vector(fused)

    def state_activity_score(self, c_base: torch.Tensor) -> torch.Tensor:
        c_base = c_base.reshape(-1).to(dtype=torch.float32)
        return self._normalize_vector(c_base.abs())

    def state_attack_alignment(self, c_base: torch.Tensor) -> torch.Tensor:
        c_base = c_base.reshape(-1).to(dtype=torch.float32)
        weighted_attack = (
            self.weighted_H.to(dtype=torch.float32) @ c_base.reshape(-1, 1)
        ).reshape(-1)
        attack_norm = float(torch.linalg.norm(weighted_attack).item())
        if attack_norm <= 1e-12:
            return self._normalize_vector(self.state_importance)

        column_norm = self.state_importance.to(dtype=torch.float32).clamp(min=1e-8)
        alignment = (
            self.weighted_H.T.to(dtype=torch.float32) @ weighted_attack
        ) / (column_norm * attack_norm)
        alignment = alignment.clamp(min=0.0)
        return self._normalize_vector(alignment)

    def state_topology_proximity(
        self,
        c_base: torch.Tensor,
        diffusion_steps: int = 1,
        diffusion_alpha: float = 0.25,
    ) -> torch.Tensor:
        support_seed = self.state_activity_score(c_base)
        if float(support_seed.max().item()) <= 0.0:
            return self.state_physical_score()

        steps = max(0, int(diffusion_steps))
        alpha = float(min(max(diffusion_alpha, 0.0), 1.0))
        proximity = support_seed.clone()
        sym_adj = 0.5 * (self.adjacency + self.adjacency.T)
        if steps > 0 and alpha > 0.0:
            for _ in range(steps):
                proximity = (1.0 - alpha) * proximity + alpha * (sym_adj @ proximity)
                proximity = self._normalize_vector(proximity)
        return self._normalize_vector(0.65 * support_seed + 0.35 * proximity)

    def state_physical_saliency(
        self,
        c_base: torch.Tensor,
        diffusion_steps: int = 1,
        diffusion_alpha: float = 0.25,
    ) -> torch.Tensor:
        structural = self.state_physical_score()
        activity = self.state_activity_score(c_base)
        if float(activity.max().item()) <= 0.0:
            return structural

        alignment = self.state_attack_alignment(c_base)
        proximity = self.state_topology_proximity(
            c_base=c_base,
            diffusion_steps=diffusion_steps,
            diffusion_alpha=diffusion_alpha,
        )
        saliency = (
            0.35 * structural
            + 0.30 * alignment
            + 0.20 * proximity
            + 0.15 * activity
        )
        return self._normalize_vector(saliency)

    def measurement_state_alignment(
        self,
        x_ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_score, state_drive = self.state_guidance_from_measurement(x_ref)
        backproj = (
            self.g_inv.to(dtype=torch.float32)
            @ (self.H.T.to(dtype=torch.float32) * self.rinv.reshape(1, -1).to(dtype=torch.float32))
        ).T
        response_score = self._normalize_vector(backproj.abs() @ state_score)
        measurement_drive = (self.H.to(dtype=torch.float32) @ state_drive.reshape(-1, 1)).reshape(-1)
        drive_profile = self._normalize_vector(
            measurement_drive.abs() / self.noise_std.to(dtype=torch.float32).clamp(min=1e-8)
        )
        return response_score, drive_profile, measurement_drive

    def measurement_physical_gate(
        self,
        x_ref: torch.Tensor,
        response_weight: float = 0.60,
        effectiveness_weight: float = 0.40,
    ) -> torch.Tensor:
        response_score, _, _ = self.measurement_state_alignment(x_ref)
        effectiveness = self._normalize_vector(self.measurement_effectiveness)
        total_weight = max(float(response_weight), 0.0) + max(
            float(effectiveness_weight), 0.0
        )
        if total_weight <= 1e-12:
            return self._normalize_vector(response_score)
        gate_score = (
            max(float(response_weight), 0.0) * response_score
            + max(float(effectiveness_weight), 0.0) * effectiveness
        ) / total_weight
        return self._normalize_vector(gate_score)

    def measurement_saliency(self, x_ref: torch.Tensor) -> torch.Tensor:
        x_ref = x_ref.reshape(-1).to(dtype=torch.float32)
        activity_score = self._normalize_vector(
            x_ref.abs() / self.noise_std.to(dtype=torch.float32).clamp(min=1e-8)
        )
        effectiveness = self._normalize_vector(self.measurement_effectiveness)
        fisher = self._normalize_vector(self.weighted_leverage)
        topology = self._normalize_vector(self.measurement_topology_score)
        state_smoothness = self._normalize_vector(self.measurement_state_smoothness)
        projection_locality = self._normalize_vector(self.measurement_projection_locality)
        saliency = (
            0.28 * effectiveness
            + 0.22 * fisher
            + 0.18 * activity_score
            + 0.15 * state_smoothness
            + 0.10 * projection_locality
            + 0.07 * topology
        )
        return self._normalize_vector(saliency)

    def _measurement_seed_score(self, x_ref: torch.Tensor) -> torch.Tensor:
        return self.measurement_saliency(x_ref)

    def measurement_region_direction_templates(
        self,
        region: torch.Tensor,
        x_ref: torch.Tensor,
    ) -> list[torch.Tensor]:
        region = region.reshape(-1).long()
        if region.numel() == 0:
            return []

        x_ref = x_ref.reshape(-1).to(dtype=torch.float32)
        saliency = self.measurement_saliency(x_ref)
        activity = self._normalize_vector(
            x_ref.abs() / self.noise_std.to(dtype=torch.float32).clamp(min=1e-8)
        )
        effectiveness = self._normalize_vector(self.measurement_effectiveness)
        fisher = self._normalize_vector(self.weighted_leverage)
        topology = self._normalize_vector(self.measurement_topology_score)
        state_smoothness = self._normalize_vector(self.measurement_state_smoothness)
        projection_locality = self._normalize_vector(self.measurement_projection_locality)
        sign = torch.sign(x_ref)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)

        local_adj = 0.5 * (
            self.measurement_adjacency[region][:, region]
            + self.measurement_adjacency[region][:, region].T
        )
        signed_saliency = saliency[region] * sign[region]
        hybrid_effective = (
            0.55 * effectiveness + 0.30 * fisher + 0.15 * activity
        )[region] * sign[region]
        topology_guided = (
            0.55 * topology + 0.25 * state_smoothness + 0.20 * saliency
        )[region] * sign[region]
        activity_guided = activity[region] * sign[region]
        smooth_guided = (
            0.60 * state_smoothness + 0.40 * projection_locality
        )[region] * sign[region]
        mixed_guided = 0.55 * signed_saliency + 0.25 * topology_guided + 0.20 * smooth_guided

        templates = [
            signed_saliency.reshape(1, -1),
            hybrid_effective.reshape(1, -1),
            topology_guided.reshape(1, -1),
            activity_guided.reshape(1, -1),
            smooth_guided.reshape(1, -1),
            mixed_guided.reshape(1, -1),
        ]
        for base_vec in (signed_saliency, topology_guided):
            diffused_vec = local_adj @ base_vec
            if float(torch.linalg.norm(diffused_vec).item()) > 1e-12:
                templates.append(diffused_vec.reshape(1, -1))
        return [
            template
            for template in templates
            if float(torch.linalg.norm(template).item()) > 1e-12
        ]

    def measurement_region_basis_vectors(
        self,
        region: torch.Tensor,
        x_ref: torch.Tensor,
        max_vectors: int = 6,
    ) -> list[torch.Tensor]:
        region = region.reshape(-1).long()
        region_dim = int(region.numel())
        if region_dim <= 0 or max_vectors <= 0:
            return []

        saliency = self.measurement_saliency(x_ref)[region].to(dtype=torch.float32)
        weight = 0.45 + 0.55 * saliency

        local_adj = 0.5 * (
            self.measurement_adjacency[region][:, region]
            + self.measurement_adjacency[region][:, region].T
        )
        local_adj = local_adj.to(dtype=torch.float32)
        local_deg = local_adj.sum(dim=1)
        local_laplacian = torch.diag(local_deg) - local_adj
        spectral_vals, spectral_vecs = torch.linalg.eigh(local_laplacian)

        projected_unit = self.measurement_region_projection(
            region=region,
            delta_m_region=torch.eye(region_dim, dtype=torch.float32),
            channel_weights=None,
            shaping_strength=0.0,
        )
        projection_gram = projected_unit @ projected_unit.T
        projection_gram = 0.5 * (projection_gram + projection_gram.T)
        proj_vals, proj_vecs = torch.linalg.eigh(projection_gram)

        basis: list[torch.Tensor] = []

        def append_unique(candidate: torch.Tensor) -> None:
            if len(basis) >= max_vectors:
                return
            candidate = candidate.reshape(1, -1).to(dtype=torch.float32)
            cand_norm = float(torch.linalg.norm(candidate).item())
            if cand_norm <= 1e-12:
                return
            candidate = candidate / cand_norm
            for existing in basis:
                cosine = float(torch.abs((candidate * existing).sum()).item())
                if cosine >= 0.98:
                    return
            basis.append(candidate)

        append_unique((weight * torch.ones_like(weight)).reshape(1, -1))
        for idx in range(region_dim - 1, max(-1, region_dim - 1 - max_vectors), -1):
            append_unique((proj_vecs[:, idx] * weight).reshape(1, -1))
        for idx in range(min(region_dim, max_vectors)):
            append_unique((spectral_vecs[:, idx] * weight).reshape(1, -1))
        return basis[:max_vectors]

    def state_region_direction_templates(
        self,
        region: torch.Tensor,
        c_base: torch.Tensor,
    ) -> list[torch.Tensor]:
        region = region.reshape(-1).long()
        if region.numel() == 0:
            return []

        c_base = c_base.reshape(-1).to(dtype=torch.float32)
        local_base = c_base[region]
        sign = torch.sign(local_base)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        activity = self._normalize_vector(local_base.abs())
        physical = self.state_physical_saliency(c_base)[region].to(dtype=torch.float32)
        structural = self.state_physical_score()[region].to(dtype=torch.float32)
        alignment = self.state_attack_alignment(c_base)[region].to(dtype=torch.float32)
        proximity = self.state_topology_proximity(c_base)[region].to(dtype=torch.float32)
        importance = self._normalize_vector(self.state_importance)[region].to(dtype=torch.float32)
        flow_support = self._normalize_vector(self.state_flow_support)[region].to(dtype=torch.float32)
        corridor = self._normalize_vector(self.state_corridor_centrality)[region].to(dtype=torch.float32)
        local_adj = 0.5 * (self.adjacency[region][:, region] + self.adjacency[region][:, region].T)
        local_adj = local_adj.to(dtype=torch.float32)

        backbone = physical * sign
        gain_guided = (0.40 * importance + 0.35 * alignment + 0.25 * flow_support) * sign
        corridor_guided = (0.45 * proximity + 0.30 * corridor + 0.25 * structural) * sign
        activity_guided = activity * sign
        mixed_guided = (
            0.40 * backbone
            + 0.30 * gain_guided
            + 0.20 * corridor_guided
            + 0.10 * activity_guided
        )

        templates = [
            backbone.reshape(1, -1),
            gain_guided.reshape(1, -1),
            corridor_guided.reshape(1, -1),
            activity_guided.reshape(1, -1),
            mixed_guided.reshape(1, -1),
        ]
        for base_vec in (backbone, mixed_guided):
            diffused_vec = local_adj @ base_vec
            if float(torch.linalg.norm(diffused_vec).item()) > 1e-12:
                templates.append(diffused_vec.reshape(1, -1))
        return [
            template
            for template in templates
            if float(torch.linalg.norm(template).item()) > 1e-12
        ]

    def state_region_basis_vectors(
        self,
        region: torch.Tensor,
        c_base: torch.Tensor,
        max_vectors: int = 6,
    ) -> list[torch.Tensor]:
        region = region.reshape(-1).long()
        region_dim = int(region.numel())
        if region_dim <= 0 or max_vectors <= 0:
            return []

        c_base = c_base.reshape(-1).to(dtype=torch.float32)
        local_base = c_base[region]
        local_activity = self._normalize_vector(local_base.abs())
        physical = self.state_physical_saliency(c_base)[region].to(dtype=torch.float32)
        weight = 0.30 + 0.45 * physical + 0.25 * local_activity

        local_adj = 0.5 * (self.adjacency[region][:, region] + self.adjacency[region][:, region].T)
        local_adj = local_adj.to(dtype=torch.float32)
        local_deg = local_adj.sum(dim=1)
        local_laplacian = torch.diag(local_deg) - local_adj
        spectral_vals, spectral_vecs = torch.linalg.eigh(local_laplacian)

        region_h = self.H[:, region].to(dtype=torch.float32)
        region_fisher = region_h.T @ (region_h * self.rinv.reshape(-1, 1).to(dtype=torch.float32))
        region_fisher = 0.5 * (region_fisher + region_fisher.T)
        fisher_vals, fisher_vecs = torch.linalg.eigh(region_fisher)

        basis: list[torch.Tensor] = []

        def append_unique(candidate: torch.Tensor) -> None:
            if len(basis) >= max_vectors:
                return
            candidate = candidate.reshape(1, -1).to(dtype=torch.float32)
            cand_norm = float(torch.linalg.norm(candidate).item())
            if cand_norm <= 1e-12:
                return
            candidate = candidate / cand_norm
            for existing in basis:
                cosine = float(torch.abs((candidate * existing).sum()).item())
                if cosine >= 0.98:
                    return
            basis.append(candidate)

        sign = torch.sign(local_base)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        append_unique((weight * sign).reshape(1, -1))
        for template in self.state_region_direction_templates(region=region, c_base=c_base):
            append_unique(template * weight.reshape(1, -1))
        for idx in range(region_dim - 1, max(-1, region_dim - 1 - max_vectors), -1):
            append_unique((fisher_vecs[:, idx] * weight).reshape(1, -1))
        for idx in range(min(region_dim, max_vectors)):
            append_unique((spectral_vecs[:, idx] * weight).reshape(1, -1))
        return basis[:max_vectors]

    def _diffuse_measurement_seed_score(
        self,
        seed_score: torch.Tensor,
        diffusion_steps: int,
        diffusion_alpha: float,
    ) -> torch.Tensor:
        steps = max(0, int(diffusion_steps))
        alpha = float(min(max(diffusion_alpha, 0.0), 1.0))
        score = self._normalize_vector(seed_score)
        if steps <= 0 or alpha <= 0.0:
            return score

        sym_adj = 0.5 * (self.measurement_adjacency + self.measurement_adjacency.T)
        for _ in range(steps):
            score = (1.0 - alpha) * score + alpha * (sym_adj @ score)
            score = self._normalize_vector(score)
        return score

    def enumerate_measurement_candidate_regions(
        self,
        x_ref: torch.Tensor,
        region_size: int,
        anchor_size: int = 2,
        num_candidates: int = 4,
        anchor_pool_size: int = 6,
        diffusion_steps: int = 2,
        diffusion_alpha: float = 0.35,
        gate_score: torch.Tensor | None = None,
        gate_ratio: float = 1.0,
    ) -> list[torch.Tensor]:
        region_size = max(1, min(int(region_size), self.n_measurements))
        anchor_size = max(1, min(int(anchor_size), region_size))
        num_candidates = max(1, int(num_candidates))
        seed_score = self.measurement_saliency(x_ref)
        diffused_score = self._diffuse_measurement_seed_score(
            seed_score=seed_score,
            diffusion_steps=diffusion_steps,
            diffusion_alpha=diffusion_alpha,
        )
        info_seed_score = self._normalize_vector(
            0.45 * self._normalize_vector(self.measurement_effectiveness)
            + 0.25 * self._normalize_vector(self.weighted_leverage)
            + 0.20 * self._normalize_vector(self.measurement_state_smoothness)
            + 0.10 * self._normalize_vector(self.measurement_projection_locality)
        )
        allowed_mask = None
        gate_ratio = min(max(float(gate_ratio), 0.0), 1.0)
        if gate_score is not None:
            gate_score = self._normalize_vector(gate_score)
        if gate_score is not None and gate_ratio < 1.0:
            keep_dim = max(
                region_size,
                anchor_size,
                int(round(gate_ratio * float(self.n_measurements))),
            )
            keep_dim = max(1, min(keep_dim, int(self.n_measurements)))
            keep_idx = torch.topk(gate_score, k=keep_dim).indices
            allowed_mask = torch.zeros(self.n_measurements, dtype=torch.bool)
            allowed_mask[keep_idx] = True
            neg_inf = torch.full_like(seed_score, -1.0)
            seed_score = torch.where(allowed_mask, seed_score, neg_inf)
            diffused_score = torch.where(allowed_mask, diffused_score, neg_inf)
            info_seed_score = torch.where(allowed_mask, info_seed_score, neg_inf)

        if allowed_mask is not None:
            allowed_count = int(allowed_mask.sum().item())
        else:
            allowed_count = int(self.n_measurements)
        pool_size = max(anchor_size, min(int(anchor_pool_size), allowed_count))
        anchor_pool = torch.topk(seed_score, k=pool_size).indices.tolist()
        diffused_pool = torch.topk(diffused_score, k=pool_size).indices.tolist()
        info_pool = torch.topk(info_seed_score, k=pool_size).indices.tolist()
        if not anchor_pool:
            anchor_pool = [0]

        candidate_regions: list[torch.Tensor] = []
        seen: set[tuple[int, ...]] = set()

        def add_region(region: torch.Tensor) -> None:
            key = tuple(int(i) for i in region.tolist())
            if key in seen:
                return
            region_set = set(key)
            for existing_region in candidate_regions:
                existing_set = set(int(i) for i in existing_region.tolist())
                overlap = len(region_set & existing_set) / max(
                    1,
                    min(len(region_set), len(existing_set)),
                )
                if overlap >= 0.85:
                    return
            seen.add(key)
            candidate_regions.append(region)

        add_region(
            self._expand_region(
                anchors=anchor_pool[:anchor_size],
                priority_score=seed_score,
                region_size=region_size,
                adjacency_matrix=self.measurement_adjacency,
                importance_values=self.measurement_effectiveness,
                allowed_mask=allowed_mask,
            )
        )
        add_region(
            self._expand_region(
                anchors=diffused_pool[:anchor_size],
                priority_score=diffused_score,
                region_size=region_size,
                adjacency_matrix=self.measurement_adjacency,
                importance_values=self.measurement_effectiveness,
                priority_weight=0.35,
                allowed_mask=allowed_mask,
            )
        )
        add_region(
            self._expand_region(
                anchors=info_pool[:anchor_size],
                priority_score=info_seed_score,
                region_size=region_size,
                adjacency_matrix=self.measurement_adjacency,
                importance_values=self.measurement_effectiveness,
                priority_weight=0.40,
                allowed_mask=allowed_mask,
            )
        )

        for primary in anchor_pool:
            if len(candidate_regions) >= num_candidates:
                break
            anchor_set = [int(primary)]
            neighbor_rank = torch.argsort(
                self.measurement_adjacency[primary]
                * (0.60 * diffused_score + 0.40 * info_seed_score),
                descending=True,
            ).tolist()
            for idx in neighbor_rank:
                idx = int(idx)
                if idx == primary or idx in anchor_set:
                    continue
                anchor_set.append(idx)
                if len(anchor_set) >= anchor_size:
                    break
            add_region(
                self._expand_region(
                    anchors=anchor_set[:anchor_size],
                    priority_score=seed_score,
                    region_size=region_size,
                    adjacency_matrix=self.measurement_adjacency,
                    importance_values=self.measurement_effectiveness,
                    allowed_mask=allowed_mask,
                )
            )

        for primary in info_pool:
            if len(candidate_regions) >= num_candidates:
                break
            anchor_set = [int(primary)]
            neighbor_rank = torch.argsort(
                self.measurement_adjacency[primary] * info_seed_score,
                descending=True,
            ).tolist()
            for idx in neighbor_rank:
                idx = int(idx)
                if idx == primary or idx in anchor_set:
                    continue
                anchor_set.append(idx)
                if len(anchor_set) >= anchor_size:
                    break
            add_region(
                self._expand_region(
                    anchors=anchor_set[:anchor_size],
                    priority_score=info_seed_score,
                    region_size=region_size,
                    adjacency_matrix=self.measurement_adjacency,
                    importance_values=self.measurement_effectiveness,
                    priority_weight=0.45,
                    allowed_mask=allowed_mask,
                )
            )

        for primary in diffused_pool:
            if len(candidate_regions) >= num_candidates:
                break
            add_region(
                self._expand_region(
                    anchors=[int(primary)],
                    priority_score=diffused_score,
                    region_size=region_size,
                    adjacency_matrix=self.measurement_adjacency,
                    importance_values=self.measurement_effectiveness,
                    priority_weight=0.40,
                    allowed_mask=allowed_mask,
                )
            )

        if not candidate_regions:
            candidate_regions.append(
                self._expand_region(
                    anchors=[int(anchor_pool[0])],
                    priority_score=seed_score,
                    region_size=region_size,
                    adjacency_matrix=self.measurement_adjacency,
                    importance_values=self.measurement_effectiveness,
                    allowed_mask=allowed_mask,
                )
            )
        return candidate_regions[:num_candidates]

    def build_region(
        self,
        c_base: torch.Tensor,
        region_size: int,
        anchor_size: int = 2,
    ) -> torch.Tensor:
        region_size = max(1, min(int(region_size), self.n_states))
        anchor_size = max(1, min(int(anchor_size), region_size))
        seed_score = self._seed_score(c_base)
        anchors = torch.topk(seed_score, k=anchor_size).indices.tolist()
        return self._expand_region(
            anchors=anchors,
            priority_score=seed_score,
            region_size=region_size,
        )

    def build_state_region_from_priority(
        self,
        priority_score: torch.Tensor,
        region_size: int,
        anchor_size: int = 2,
        importance_values: torch.Tensor | None = None,
        priority_weight: float = 0.25,
        allowed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        region_size = max(1, min(int(region_size), self.n_states))
        anchor_size = max(1, min(int(anchor_size), region_size))
        priority_score = self._normalize_vector(priority_score)
        if float(priority_score.max().item()) <= 0.0:
            priority_score = self.state_physical_score()
        if allowed_mask is not None:
            allowed_mask = allowed_mask.reshape(-1).to(dtype=torch.bool)
            masked_priority = torch.where(
                allowed_mask,
                priority_score,
                torch.full_like(priority_score, -1.0),
            )
            anchors = torch.topk(masked_priority, k=anchor_size).indices.tolist()
        else:
            anchors = torch.topk(priority_score, k=anchor_size).indices.tolist()
        return self._expand_region(
            anchors=anchors,
            priority_score=priority_score,
            region_size=region_size,
            adjacency_matrix=self.adjacency,
            importance_values=self.state_importance
            if importance_values is None
            else importance_values,
            priority_weight=priority_weight,
            allowed_mask=allowed_mask,
        )

    def enumerate_candidate_regions(
        self,
        c_base: torch.Tensor,
        region_size: int,
        anchor_size: int = 2,
        num_candidates: int = 4,
        anchor_pool_size: int = 6,
        probabilistic_prior: bool = False,
        prior_temperature: float = 1.0,
        prior_uniform_mixing: float = 0.0,
        generator: torch.Generator | None = None,
        proposal_mode: str = "default",
        diffusion_steps: int = 2,
        diffusion_alpha: float = 0.35,
        flow_weight: float = 0.30,
        corridor_weight: float = 0.25,
        seed_score: torch.Tensor | None = None,
        allowed_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        region_size = max(1, min(int(region_size), self.n_states))
        anchor_size = max(1, min(int(anchor_size), region_size))
        num_candidates = max(1, int(num_candidates))
        if seed_score is None:
            seed_score = self._seed_score(c_base)
        else:
            seed_score = self._normalize_vector(seed_score)
        if allowed_mask is not None:
            allowed_mask = allowed_mask.reshape(-1).to(dtype=torch.bool)
            masked_seed_score = torch.where(
                allowed_mask,
                seed_score,
                torch.full_like(seed_score, -1.0),
            )
        else:
            masked_seed_score = seed_score

        if allowed_mask is not None:
            allowed_count = int(allowed_mask.sum().item())
        else:
            allowed_count = int(self.n_states)
        pool_size = max(anchor_size, min(int(anchor_pool_size), allowed_count))
        greedy_pool = torch.topk(masked_seed_score, k=pool_size).indices.tolist()
        if probabilistic_prior:
            sampled_pool = self._sample_index_pool(
                score=self._normalize_vector(
                    torch.where(
                        allowed_mask,
                        seed_score,
                        torch.zeros_like(seed_score),
                    )
                )
                if allowed_mask is not None
                else seed_score,
                pool_size=pool_size,
                generator=generator,
                temperature=prior_temperature,
                uniform_mixing=prior_uniform_mixing,
            )
            anchor_pool = []
            for idx in greedy_pool[:anchor_size]:
                if int(idx) not in anchor_pool:
                    anchor_pool.append(int(idx))
            for idx in sampled_pool:
                if int(idx) not in anchor_pool:
                    anchor_pool.append(int(idx))
            for idx in greedy_pool:
                if len(anchor_pool) >= pool_size:
                    break
                if int(idx) not in anchor_pool:
                    anchor_pool.append(int(idx))
        else:
            anchor_pool = greedy_pool
        if not anchor_pool:
            anchor_pool = [0]

        candidate_regions: list[torch.Tensor] = []
        seen: set[tuple[int, ...]] = set()
        candidate_anchor_sets: list[list[int]] = []

        def add_region(region: torch.Tensor) -> None:
            key = tuple(int(i) for i in region.tolist())
            if key in seen:
                return
            seen.add(key)
            candidate_regions.append(region)

        if proposal_mode == "hierarchical_corridor":
            diffused_score = self._diffuse_seed_score(
                seed_score=seed_score,
                diffusion_steps=diffusion_steps,
                diffusion_alpha=diffusion_alpha,
            )
            corridor_score = self._corridor_seed_score(
                seed_score=seed_score,
                diffusion_steps=diffusion_steps,
                diffusion_alpha=diffusion_alpha,
                flow_weight=flow_weight,
                corridor_weight=corridor_weight,
            )
            diffused_pool = torch.argsort(diffused_score, descending=True).tolist()
            corridor_pool = torch.argsort(corridor_score, descending=True).tolist()

            add_region(
                self._expand_region(
                        anchors=anchor_pool[:anchor_size],
                        priority_score=seed_score,
                        region_size=region_size,
                        allowed_mask=allowed_mask,
                    )
                )

            if diffused_pool and len(candidate_regions) < num_candidates:
                add_region(
                    self._expand_region(
                        anchors=[int(idx) for idx in diffused_pool[:anchor_size]],
                        priority_score=diffused_score,
                        region_size=region_size,
                        priority_weight=0.35,
                        allowed_mask=allowed_mask,
                    )
                )

            if anchor_pool and corridor_pool and len(candidate_regions) < num_candidates:
                src = int(anchor_pool[0])
                dst = next((int(idx) for idx in corridor_pool if int(idx) != src), src)
                path = self._strongest_path(src=src, dst=dst)
                add_region(
                    self._expand_region(
                        anchors=path,
                        priority_score=corridor_score,
                        region_size=region_size,
                        priority_weight=0.45,
                        allowed_mask=allowed_mask,
                    )
                )

            if anchor_pool and len(candidate_regions) < num_candidates:
                primary = int(anchor_pool[0])
                anchor_set = [primary]
                mix_rank = torch.argsort(
                    (self.adjacency[primary] + self.adjacency[:, primary]) * corridor_score,
                    descending=True,
                ).tolist()
                for idx in mix_rank:
                    idx = int(idx)
                    if idx == primary or idx in anchor_set:
                        continue
                    anchor_set.append(idx)
                    if len(anchor_set) >= anchor_size:
                        break
                add_region(
                    self._expand_region(
                        anchors=anchor_set[:anchor_size],
                        priority_score=corridor_score,
                        region_size=region_size,
                        priority_weight=0.40,
                        allowed_mask=allowed_mask,
                    )
                )

        candidate_anchor_sets.append(anchor_pool[:anchor_size])
        for primary in anchor_pool[:num_candidates]:
            anchor_set = [int(primary)]
            if anchor_size > 1:
                neighbor_rank = torch.argsort(
                    self.adjacency[primary] * seed_score, descending=True
                ).tolist()
                for idx in neighbor_rank:
                    idx = int(idx)
                    if idx == primary or idx in anchor_set:
                        continue
                    anchor_set.append(idx)
                    if len(anchor_set) >= anchor_size:
                        break
            candidate_anchor_sets.append(anchor_set[:anchor_size])

        if len(anchor_pool) >= anchor_size:
            for offset in range(1, min(num_candidates, len(anchor_pool) - 1) + 1):
                candidate_anchor_sets.append(
                    [int(anchor_pool[0]), int(anchor_pool[offset])][:anchor_size]
                )

        for anchors in candidate_anchor_sets:
            if len(candidate_regions) >= num_candidates:
                break
            if not anchors:
                continue
            add_region(
                self._expand_region(
                    anchors=anchors,
                    priority_score=seed_score,
                    region_size=region_size,
                    allowed_mask=allowed_mask,
                )
            )

        if not candidate_regions:
            candidate_regions.append(
                self._expand_region(
                    anchors=[int(anchor_pool[0])],
                    priority_score=seed_score,
                    region_size=region_size,
                    allowed_mask=allowed_mask,
                )
            )
        return candidate_regions

    def build_measurement_region_from_priority(
        self,
        priority_score: torch.Tensor,
        region_size: int,
        anchor_size: int = 2,
        importance_values: torch.Tensor | None = None,
        priority_weight: float = 0.35,
        allowed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        region_size = max(1, min(int(region_size), self.n_measurements))
        anchor_size = max(1, min(int(anchor_size), region_size))
        priority_score = self._normalize_vector(priority_score)
        if float(priority_score.max().item()) <= 0.0:
            priority_score = self._normalize_vector(self.measurement_effectiveness)
        if allowed_mask is not None:
            allowed_mask = allowed_mask.reshape(-1).to(dtype=torch.bool)
            masked_priority = torch.where(
                allowed_mask,
                priority_score,
                torch.full_like(priority_score, -1.0),
            )
            anchors = torch.topk(masked_priority, k=anchor_size).indices.tolist()
        else:
            anchors = torch.topk(priority_score, k=anchor_size).indices.tolist()
        return self._expand_region(
            anchors=anchors,
            priority_score=priority_score,
            region_size=region_size,
            adjacency_matrix=self.measurement_adjacency,
            importance_values=self.measurement_effectiveness
            if importance_values is None
            else importance_values,
            priority_weight=priority_weight,
            allowed_mask=allowed_mask,
        )

    def build_channel_weights(
        self,
        injection_scale: float = 0.82,
        flow_scale: float = 1.0,
        leverage_suppression: float = 0.30,
        min_channel_scale: float = 0.60,
    ) -> torch.Tensor:
        weights = torch.full(
            (self.n_measurements,),
            float(flow_scale),
            dtype=torch.float32,
        )
        inj_mask = self.measurement_type_ids == 1
        weights[inj_mask] = float(injection_scale)

        if float(self.weighted_leverage.max().item()) > 0.0 and leverage_suppression > 0.0:
            leverage_norm = self.weighted_leverage / self.weighted_leverage.max().clamp(
                min=1e-8
            )
            weights = weights * (1.0 - float(leverage_suppression) * leverage_norm)
        return weights.clamp(min=float(min_channel_scale), max=1.0)

    def measurement_region_budget_prior(
        self,
        region: torch.Tensor,
        x_ref: torch.Tensor,
    ) -> float:
        region = region.reshape(-1).long()
        if region.numel() == 0:
            return 0.0

        seed_score = self.measurement_saliency(x_ref)
        saliency_score = float(seed_score[region].mean().item())
        smoothness_score = float(
            (
                0.55 * self._normalize_vector(self.measurement_state_smoothness)
                + 0.45 * self._normalize_vector(self.measurement_projection_locality)
            )[region].mean().item()
        )
        activity_score = float(
            self._normalize_vector(
                x_ref.reshape(-1).abs()
                / self.noise_std.to(dtype=torch.float32).clamp(min=1e-8)
            )[region].mean().item()
        )
        cohesion_score = float(
            self.measurement_adjacency[region][:, region].mean().item()
        )
        weighted_region_h = self.weighted_H[region].to(dtype=torch.float32)
        info_mat = torch.eye(int(region.numel()), dtype=torch.float32) + (
            weighted_region_h @ self.g_inv.to(dtype=torch.float32) @ weighted_region_h.T
        )
        sign, logabsdet = torch.linalg.slogdet(info_mat)
        info_score = 0.0
        if float(sign.item()) > 0.0:
            info_score = 1.0 - float(
                torch.exp(-logabsdet / max(1.0, float(region.numel()))).item()
            )
        score = (
            0.40 * info_score
            + 0.25 * saliency_score
            + 0.15 * smoothness_score
            + 0.10 * activity_score
            + 0.10 * cohesion_score
        )
        return float(max(0.0, min(1.0, score)))

    def region_budget_prior(
        self,
        region: torch.Tensor,
        c_base: torch.Tensor | None = None,
    ) -> float:
        region = region.reshape(-1).long()
        if region.numel() == 0:
            return 0.0

        importance_score = float(
            self._normalize_vector(self.state_importance)[region].mean().item()
        )
        boundary_penalty = self.state_region_boundary_penalty(region)

        # Simple state-region prior used by the mainline:
        # maximize average physical gain inside the region while discouraging
        # regions with large boundary leakage (conductance-style penalty).
        prior = importance_score - 0.35 * boundary_penalty
        return float(max(0.0, min(1.0, prior)))

    def state_sparse_efficiency(
        self,
        beta: float = 0.5,
    ) -> torch.Tensor:
        beta = max(float(beta), 0.0)
        support = self.state_measurement_support_count.to(dtype=torch.float32).clamp(min=1.0)
        efficiency = self.state_importance.to(dtype=torch.float32) / support.pow(beta)
        return self._normalize_vector(efficiency)

    def state_region_boundary_penalty(self, region: torch.Tensor) -> float:
        region = region.reshape(-1).long()
        if region.numel() == 0:
            return 1.0
        sym_adj = 0.5 * (self.adjacency + self.adjacency.T)
        region_mask = torch.zeros(self.n_states, dtype=torch.bool)
        region_mask[region] = True
        region_volume = float(sym_adj[region].sum().item())
        if region_volume <= 1e-12:
            return 1.0
        boundary_penalty = float(sym_adj[region][:, ~region_mask].sum().item()) / region_volume
        return float(max(0.0, min(1.0, boundary_penalty)))

    def state_region_measurement_support_count(self, region: torch.Tensor) -> int:
        region = region.reshape(-1).long()
        if region.numel() == 0:
            return 0
        region_mass = self.H[:, region].abs().sum(dim=1)
        support_count = int((region_mass > 1e-10).sum().item())
        return support_count

    def state_region_measurement_support_ratio(self, region: torch.Tensor) -> float:
        support_count = self.state_region_measurement_support_count(region)
        return float(support_count / max(1, int(self.n_measurements)))

    def measurement_region_projection(
        self,
        region: torch.Tensor,
        delta_m_region: torch.Tensor,
        channel_weights: torch.Tensor | None = None,
        shaping_strength: float = 0.0,
    ) -> torch.Tensor:
        delta_m_region = delta_m_region.to(dtype=torch.float32)
        if delta_m_region.ndim == 1:
            delta_m_region = delta_m_region.unsqueeze(0)
        masked_delta = torch.zeros(
            delta_m_region.shape[0],
            self.n_measurements,
            dtype=delta_m_region.dtype,
            device=delta_m_region.device,
        )
        masked_delta[:, region] = delta_m_region
        if (
            channel_weights is not None
            and channel_weights.numel() == self.n_measurements
            and shaping_strength > 0.0
        ):
            shaping_strength = float(min(max(shaping_strength, 0.0), 1.0))
            weights = channel_weights.to(
                device=masked_delta.device,
                dtype=masked_delta.dtype,
            ).reshape(1, -1)
            blend = (1.0 - shaping_strength) + shaping_strength * weights
            masked_delta = masked_delta * blend
        return self.project_measurement_to_col_space(masked_delta)

    def region_projection(
        self,
        region: torch.Tensor,
        delta_c_region: torch.Tensor,
        channel_weights: torch.Tensor | None = None,
        shaping_strength: float = 0.0,
    ) -> torch.Tensor:
        region_H = self.H[:, region].to(
            device=delta_c_region.device,
            dtype=delta_c_region.dtype,
        )
        delta_z = delta_c_region @ region_H.T
        if (
            channel_weights is None
            or channel_weights.numel() != self.n_measurements
            or shaping_strength <= 0.0
        ):
            return delta_z

        shaping_strength = float(min(max(shaping_strength, 0.0), 1.0))
        weights = channel_weights.to(
            device=delta_z.device,
            dtype=delta_z.dtype,
        ).reshape(1, -1)
        blend = (1.0 - shaping_strength) + shaping_strength * weights
        shaped_delta = delta_z * blend

        rinv = self.rinv.to(device=delta_z.device, dtype=delta_z.dtype).reshape(1, -1)
        region_gram = region_H.T @ (region_H * rinv.reshape(-1, 1))
        region_gram_pinv = torch.linalg.pinv(region_gram)
        coeff = (shaped_delta * rinv) @ region_H @ region_gram_pinv
        return coeff @ region_H.T

    def summarize_measurement_region(self, region: torch.Tensor, x_ref: torch.Tensor | None = None) -> dict:
        region = region.reshape(-1).long()
        if x_ref is None:
            x_ref = torch.ones(self.n_measurements, dtype=torch.float32)
        saliency = self._measurement_seed_score(x_ref)
        assoc_bus_ids = set()
        assoc_branch_ids = set()
        for idx in region.tolist():
            bus_values = (
                self.measurement_bus_ids[idx],
                self.measurement_from_bus_ids[idx],
                self.measurement_to_bus_ids[idx],
            )
            for value in bus_values:
                bus_id = int(round(float(value.item())))
                if bus_id > 0:
                    assoc_bus_ids.add(bus_id)
            branch_id = int(round(float(self.measurement_branch_ids[idx].item())))
            if branch_id > 0:
                assoc_branch_ids.add(branch_id)
        return {
            "region_size": int(region.numel()),
            "region_measurement_ids": [int(i) for i in region.tolist()],
            "region_measurement_types": [
                int(self.measurement_type_ids[i].item())
                for i in region.tolist()
            ],
            "region_bus_ids": sorted(assoc_bus_ids),
            "region_branch_ids": sorted(assoc_branch_ids),
            "region_measurement_saliency_mean": float(saliency[region].mean().item()),
            "region_measurement_effectiveness_mean": float(
                self.measurement_effectiveness[region].mean().item()
            ),
            "region_measurement_stealth_mean": float(
                self.measurement_stealth[region].mean().item()
            ),
            "region_measurement_topology_mean": float(
                self.measurement_topology_score[region].mean().item()
            ),
            "graph_source": self.graph_source,
            "region_space": "measurement",
        }

    def summarize_region(
        self,
        region: torch.Tensor,
        region_space: str = "state",
        x_ref: torch.Tensor | None = None,
        c_base: torch.Tensor | None = None,
    ) -> dict:
        if str(region_space).lower() == "measurement":
            return self.summarize_measurement_region(region=region, x_ref=x_ref)
        region = region.reshape(-1).long()
        boundary_penalty = self.state_region_boundary_penalty(region)
        support_count = self.state_region_measurement_support_count(region)
        support_ratio = self.state_region_measurement_support_ratio(region)
        summary = {
            "region_size": int(region.numel()),
            "region_state_ids": [int(i) for i in region.tolist()],
            "region_bus_ids": [
                int(self.state_bus_ids[i].item())
                for i in region.tolist()
                if i < self.state_bus_ids.numel()
            ],
            "region_importance_mean": float(self.state_importance[region].mean().item()),
            "region_importance_max": float(self.state_importance[region].max().item()),
            "region_flow_support_mean": float(self.state_flow_support[region].mean().item()),
            "region_corridor_centrality_mean": float(
                self.state_corridor_centrality[region].mean().item()
            ),
            "region_degree_mean": float(self.state_degree[region].mean().item())
            if self.state_degree.numel() == self.state_importance.numel()
            else 0.0,
            "region_measurement_support_count": int(support_count),
            "region_measurement_support_ratio": float(support_ratio),
            "region_boundary_penalty": float(boundary_penalty),
            "region_connectivity_score": float(1.0 - boundary_penalty),
            "graph_source": self.graph_source,
            "region_space": "state",
        }
        if c_base is not None:
            physical_saliency = self.state_physical_saliency(c_base)
            alignment = self.state_attack_alignment(c_base)
            proximity = self.state_topology_proximity(c_base)
            activity = self.state_activity_score(c_base)
            sparse_efficiency = self.state_sparse_efficiency()
            summary.update(
                {
                    "region_physical_saliency_mean": float(
                        physical_saliency[region].mean().item()
                    ),
                    "region_alignment_mean": float(alignment[region].mean().item()),
                    "region_proximity_mean": float(proximity[region].mean().item()),
                    "region_activity_mean": float(activity[region].mean().item()),
                    "region_sparse_efficiency_mean": float(
                        sparse_efficiency[region].mean().item()
                    ),
                }
            )
        return summary
