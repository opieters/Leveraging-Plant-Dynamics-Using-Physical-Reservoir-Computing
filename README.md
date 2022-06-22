# Analysis Scripts for 'Leveraging Plant Dynamics Using Physical Reservoir Computing'

This repo contains the necessary scripts to regenerate the results from the paper "Leveraging Plant Dynamics Using Physical Reservoir Computing". The data is available on [Zenodo](https://doi.org/10.5281/zenodo.4264624) and the code on [GitHub](https://github.com/opieters/Leveraging-Plant-Dynamics-Using-Physical-Reservoir-Computing).

## Installation

Install Python3.10 and the requirements in `requirements.txt` for the necessary dependencies.

## Data

There is leaf thickness and physiological data available from three experiments: a control experiment and two strawberry experiments. In each experiment, the environmental conditions of a growth chamber are modulated (light intensity,  temperature and relative humidity) and a single strawberry plant is located inside the chamber. Leaf thickness measurement clips are mounted on the plant except for the control experiment. In this case a plant is still inserted but the clips are not mounted. Physiological data of the plant is collected in all three experiments using a LI6400XT photosynthesis system.

An overview of the available parameters is included below. N/A refers to a sensor that is not calibrated and/or temperature compensated. Calibration data is available TODO

| parameter                                           | unit      | description                                        |
|-----------------------------------------------------|-----------|----------------------------------------------------|
| light_sensor_lux                                    | lux       | light intensity (humain)                           |
| leaf_thickness_1_um                                 | um        | thickness of leaf clip 1                           |
| leaf_thickness_1_nc_au                              | N/A       | thickness of leaf clip 1                           |
| leaf_temp_1_nc_au                                   | N/A       | temperature of leaf clip 1                         |
| leaf_thickness_2_um                                 | um        | thickness of leaf clip 2                           |
| leaf_thickness_2_nc_au                              | N/A       | thickness of leaf clip 2                           |
| leaf_temp_2_nc_au                                   | N/A       | temperature of leaf clip 2                         |
| leaf_thickness_3_um                                 | um        | thickness of leaf clip 3                           |
| leaf_thickness_3_nc_au                              | N/A       | thickness of leaf clip 3                           |
| leaf_temp_3_nc_au                                   | N/A       | temperature of leaf clip 3                         |
| leaf_thickness_4_um                                 | um        | thickness of leaf clip 4                           |
| leaf_thickness_4_nc_au                              | N/A       | thickness of leaf clip 4                           |
| leaf_temp_4_nc_au                                   | N/A       | temperature of leaf clip 4                         |
| leaf_thickness_5_um                                 | um        | thickness of leaf clip 5                           |
| leaf_thickness_5_nc_au                              | N/A       | thickness of leaf clip 5                           |
| leaf_temp_5_nc_au                                   | N/A       | temperature of leaf clip 5                         |
| leaf_thickness_6_um                                 | um        | thickness of leaf clip 6                           |
| leaf_thickness_6_nc_au                              | N/A       | thickness of leaf clip 6                           |
| leaf_temp_6_nc_au                                   | N/A       | temperature of leaf clip 6                         |
| leaf_thickness_7_um                                 | um        | thickness of leaf clip 7                           |
| leaf_thickness_7_nc_au                              | N/A       | raw leaf thickness measurement                     |
| leaf_temp_7_nc_au                                   | N/A       | temperature of leaf clip 7                         |
| leaf_thickness_8_um                                 | um        | thickness of leaf clip 8                           |
| leaf_thickness_8_nc_au                              | N/A       | thickness of leaf clip 8                           |
| leaf_temp_8_nc_au                                   | N/A       | temperature of leaf clip 8                         |
| ref_mon_0                                           | N/A       | monitor of the 3.3V ADC reference (board 0)        |
| ref_mon_1                                           | N/A       | monitor of the 3.3V ADC reference (board 1)        |
| ref_mon_2                                           | N/A       | monitor of the 3.3V ADC reference (board 2)        |
| ref_mon_3                                           | N/A       | monitor of the 3.3V ADC reference (board 3)        |
| soil_water_content_nc_au                            | N/A       | soil water concentration                           |
| air_temperature_C                                   | degree C  | air temperature                                    |
| relative_humidity_percent                           | %         | relative humidity                                  |
| li6400xt_external_probe_air_temperature_C           | degree C  | air temperature of external probe                  |
| li6400xt_external_probe_relative_humidity_percent   | %         | rel. humidity of external probe                    |
| li6400xt_photosynthetic_rate_umol/m2/s              | umol/m2/s | photosynthetic rate                                |
| li6400xt_stomatal_conductance_mol/m2/s              | mol/m2/s  | stomatal conductance                               |
| li6400xt_transpiration_rate_mmol/m2/s               | mmol/m2/s | transoration rate                                  |
| li6400xt_vapour_pressure_deficit_kPa                | kPa       | vapour pressure deficit                            |
| li6400xt_sample_cell_air_temperature_C              | degree C  | air temperature in the sample cell                 |
| li6400xt_leaf_temperature_C                         | degree C  | temperature of leaf inside sample cell             |
| li6400xt_ref_cell_CO2_conc_umol/mol                 | umol/mol  | CO2 concentration in the reference cell            |
| li6400xt_sample_cell_CO2_conc_umol/mol              | umol/mol  | CO2 concentration in the sample cell               |
| li6400xt_ref_cell_H2O_conc_mmol/mol                 | mmol/mol  | H2O concentration in the reference cell            |
| li6400xt_sample_cell_H2O_conc_mmol/mol              | mmol/mol  | H2O concentration in the sample cell               |
| li6400xt_ref_cell_relative_humidity_conc_percent    | %         | Rel. humidity in the reference cell                |
| li6400xt_sample_cell_relative_humidity_conc_percent | %         | Rel. humidity in the sample cell                   |
| li6400xt_PAR_inside_chamber_umol/m2/s               | umol/m2/s | PAR inside leaf chamber                            |
| li6400xt_PAR_outside_chamber_umol/m2/s              | umol/m2/s | PAR outside leaf chamber                           |
| li6400xt_air_pressure_kPa                           | kPa       | air pressure                                       |
| time                                                | time      | sample time                                        |
| train_val_test_split                                |           | data split on da day-basis, all days equal         |
| train_val_test_split2                               |           | data split on da day-basis, train focus on first 5 |
| di_2                                                |           | discard interval for +-2h from center of night     |
| di_4                                                |           | discard interval for +-4h from center of night     |
| di_6                                                |           | discard interval for +-6h from center of night     |
| di_9                                                |           | discard interval for +-9h from center of night     |
| di2_2                                               |           | discard interval for +-2h from center of night     |
| di2_4                                               |           | discard interval for +-4h from center of night     |
| di2_6                                               |           | discard interval for +-6h from center of night     |
| di2_9                                               |           | discard interval for +-9h from center of night     |

Non-calibrated variables are usually RAW ADC readout values. The ADC range is 0-3.3V, where the midpoint is at 1.65V (0x0).

`train_val_test_split` interleaves the train and test splits, such that drift in the system is automatically compensated for, while `train_val_test_split2` does not. The test data is always at the end of the analysis. `train_val_test_split` should be used with `di_`, and `train_val_test_split2` should be used with `di2_`.

Three data formats are available: `data`, `full_data` and `mini_data`. `data` was used to generate the results. It is a cropped version of `full_data` that discards part of the start of the experiment and end to remove transient effects at the start. `mini_data` is a subsampled dataset, with sample spacing of 60s (sample interval), which is useful for plotting and fast analysis.

## Results

The results are generated in a two-step process. First, the number crunching is done and the results are stored in pickle files. Then, analysis and data visualisation scripts are run to interpret them.

All graphs from the paper can be reproduced using these scripts. The advised order for running them is:

TODO

## Cite

If you use the data, please cite the Zenodo repository:
TODO

The peer-reviewed paper can be cited as follows:
TODO

## Licence 

MIT, see LICENCE file.

