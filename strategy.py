#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation (>1.5x 20 EMA)
# Uses Camarilla pivot levels from prior completed 1d for structure, 1d EMA34 for higher timeframe trend filter
# Volume confirmation ensures breakout has strong participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h.
# 1d EMA34 provides strong trend filter, reducing whipsaw while capturing major moves in both bull and bear markets.
# Camarilla R3/S3 breakouts work well when combined with volume and trend filters.

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter from prior completed 1d bar
    close_1d = df_1d['close'].values
    close_1d_shifted = np.roll(close_1d, 1)
    close_1d_shifted[0] = np.nan
    ema_34_1d = pd.Series(close_1d_shifted).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from prior completed 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = np.roll(close_1d, 1)
    close_1d_prev[0] = np.nan
    
    # Camarilla levels: R3/S3, R4/S4
    # R3 = Close + 1.1*(High-Low)/4
    # S3 = Close - 1.1*(High-Low)/4
    # R4 = Close + 1.1*(High-Low)/2
    # S4 = Close - 1.1*(High-Low)/2
    camarilla_r3 = close_1d_prev + 1.1 * (high_1d - low_1d) / 4
    camarilla_s3 = close_1d_prev - 1.1 * (high_1d - low_1d) / 4
    camarilla_r4 = close_1d_prev + 1.1 * (high_1d - low_1d) / 2
    camarilla_s4 = close_1d_prev - 1.1 * (high_1d - low_1d) / 2
    
    # Shift by 1 to use only prior completed 1d bar
    camarilla_r3_shifted = np.roll(camarilla_r3, 1)
    camarilla_s3_shifted = np.roll(camarilla_s3, 1)
    camarilla_r4_shifted = np.roll(camarilla_r4, 1)
    camarilla_s4_shifted = np.roll(camarilla_s4, 1)
    camarilla_r3_shifted[0] = np.nan
    camarilla_s3_shifted[0] = np.nan
    camarilla_r4_shifted[0] = np.nan
    camarilla_s4_shifted[0] = np.nan
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_shifted)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_shifted)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_shifted)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 + price above 1d EMA34 + volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 + price below 1d EMA34 + volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint of R3/S3 OR price crosses below 1d EMA34
            midpoint = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if not np.isnan(midpoint) and (close[i] < midpoint or close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint of R3/S3 OR price crosses above 1d EMA34
            midpoint = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if not np.isnan(midpoint) and (close[i] > midpoint or close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals