# run_single_gc.py

from multiprocessing import freeze_support

from gc_binary_pipeline.config import ClusterConfig
from gc_binary_pipeline.run_cluster import run_cluster


def main():
    cfg = ClusterConfig(
        cluster_name="NGC5272",
        file_path="golden_samples/HST_56GC/ngc5272/hlsp_hugs_hst_wfc3-uvis-acs-wfc_ngc5272_multi_v1_catalog-meth1.txt",
        mag_min=19.0,
        mag_max=21.5,

        # Debug 阶段建议先小一点
        sample_size=5000,

        apply_dr=True,
        reverse_population_assignment=False,

        # Debug 阶段建议先减少 MCMC 量
        draws=1000,
        tune=1000,
        chains=4,

        output_dir="outputs",
    )
    cfg = ClusterConfig(
        cluster_name="NGC5272_reverse",
        file_path="golden_samples/HST_56GC/ngc5272/hlsp_hugs_hst_wfc3-uvis-acs-wfc_ngc5272_multi_v1_catalog-meth1.txt",
        mag_min=19.0,
        mag_max=21.5,
        sample_size=5000,
        apply_dr=True,
        reverse_population_assignment=True,
        delta_abs_max=3.0,
        template_delta_abs_max=2.0,
        binary_primary_mag_buffer=0.85,
        draws=1000,
        tune=1000,
        chains=4,
        output_dir="outputs_6752",
    )

    cfg = ClusterConfig(
            cluster_name="NGC6752_clean090",
            file_path="golden_samples/HST_56GC/ngc6752/hlsp_hugs_hst_wfc3-uvis-acs-wfc_ngc6752_multi_v1_catalog-meth1.txt",
            mag_min=17.5,
            mag_max=21,

            sample_size=5000,
            apply_dr=True,
            reverse_population_assignment=False,

            init_prob_threshold=0.90,
            delta_abs_max=3.0,
            template_delta_abs_max=1.5,
            binary_primary_mag_buffer=0.85,

            draws=1000,
            tune=1000,
            chains=4,
            

            output_dir="outputs_6752",
        )



    result = run_cluster(cfg)
    return result


if __name__ == "__main__":
    freeze_support()
    main()
