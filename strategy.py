#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike (>2.0x 20 EMA)
# Uses Camarilla pivot levels from prior completed 1d for structure, 1d EMA34 for higher timeframe trend filter
# Volume confirmation ensures breakout has strong participation
# Discrete sizing 0.30 balances risk and fee efficiency
# Target: 75-200 total trades over 4 years = 19-50/year for 4h.
# 1d EMA34 provides strong trend filter, reducing whipsaw while capturing major moves in both bull and bear markets.
# Camarilla pivots work well when combined with volume and trend filters.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter from prior completed 1d bar
    close_1d = df_1d['close'].values
    close_1d_shifted = np.roll(close_1d, 1)
    close_1d_shifted[0] = np.nan
    ema_34_1d = pd.Series(close_1d_shifted).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Camarilla pivot levels (R3, S3) from prior completed 1d bar
    # Camarilla: R3 = H + 1.1*(H-L)/2, S3 = L - 1.1*(H-L)/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels from prior completed 1d bar
    hl_range = high_1d - low_1d
    r3 = high_1d + 1.1 * hl_range / 2
    s3 = low_1d - 1.1 * hl_range / 2
    
    # Shift by 1 to use only prior completed 1d bar
    r3_shifted = np.roll(r3, 1)
    s3_shifted = np.roll(s3, 1)
    r3_shifted[0] = np.nan
    s3_shifted[0] = np.nan
    
    # Align to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_shifted)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 + price above 1d EMA34 + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below S3 + price below 1d EMA34 + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint of R3-S3 OR price crosses below 1d EMA34
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if not np.isnan(midpoint) and (close[i] < midpoint or close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price returns to midpoint of R3-S3 OR price crosses above 1d EMA34
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if not np.isnan(midpoint) and (close[i] > midpoint or close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals