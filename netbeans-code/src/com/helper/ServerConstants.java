/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.helper;

import java.util.HashMap;

/**
 *
 * @author technowings
 */
public class ServerConstants {

    public static HashMap columnsFilter = new HashMap();

    public static final String PROJECT_DIR = ".\\";
    public static final String FEATURES_CSV = ".\\dataset\\svm\\training.csv";
    public static final String AUDIO_DATASET_DIR = ServerConstants.PROJECT_DIR + "dataset\\patient-vocal-dataset-small\\";
    public static final String MOODS_DATA_FILE = PROJECT_DIR + "moods_cached_all_Features.bin";
    public static final String MOODS_DATA_ATTR_NAMES = PROJECT_DIR + "moods_cached_all_Features_keys.bin";
    public static HashMap moods = null;
}
