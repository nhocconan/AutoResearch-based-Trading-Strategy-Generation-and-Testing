#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and chop regime filter.
Long when price breaks above 1d Camarilla R3 level AND 1d volume > 2x 20-period average AND chop < 61.8 (trending).
Short when price breaks below 1d Camarilla S3 level AND 1d volume > 2x 20-period average AND chop < 61.8.
Exit when price touches opposite Camarilla level (S3 for longs, R3 for shorts).
Uses 1d HTF for Camarilla levels, volume, and chop to avoid noise and ensure structure.
Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
Camarilla levels provide intraday support/resistance; volume spike confirms institutional interest;
chop filter ensures we trade only in trending markets (works in both bull and bear when trends exist).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels, volume, and chop (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels (R3, S3)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * range_1d * 1.1 / 4
    camarilla_s3 = close_1d - 1.1 * range_1d * 1.1 / 4
    
    # Volume spike filter: volume > 2x 20-period average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (CHOP) - values < 38.2 = trending, > 61.8 = ranging
    # CHOP = 100 * log10(sum(ATR over n) / (max(high)-min(low) over n)) / log10(n)
    atr_1d = []
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                 np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = 0
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=1).mean().values
    
    sum_atr_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = max_high_1d - min_low_1d
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop_1d = 100 * np.log10(sum_atr_1d / chop_denom) / np.log10(14)
    
    # Align all 1d indicators to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 12h price for breakout detection
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 20  # need 20 periods for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        vol_ma_val = vol_ma_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        volume_12h = prices['volume'].iloc[i]
        
        if position == 0:
            # Long: Break above R3 AND volume spike AND trending market (chop < 61.8)
            if price > r3 and volume_12h > 2.0 * vol_ma_val and chop_val < 61.8:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 AND volume spike AND trending market (chop < 61.8)
            elif price < s3 and volume_12h > 2.0 * vol_ma_val and chop_val < 61.8:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price touches opposite Camarilla level
            if position == 1 and price < s3:  # Long exit at S3
                exit_signal = True
            elif position == -1 and price > r3:  # Short exit at R3
                exit_signal = True
            else:
                exit_signal = False
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R3S3_Breakout_1dVolumeSpike_ChopFilter_LevelExit"
timeframe = "12h"
leverage = 1.0