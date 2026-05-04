#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike (>2.0x 20 EMA volume)
# Uses Camarilla levels from prior completed 1d bar for structure, 1w EMA50 for higher timeframe trend filter
# Volume confirmation ensures breakout has strong participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 30-100 total trades over 4 years = 7-25/year for 1d.
# 1w EMA50 provides strong trend filter, reducing whipsaw while capturing major moves in both bull and bear markets.
# Camarilla breakouts work well when combined with volume and trend filters.

name = "1d_Camarilla_R3S3_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter from prior completed 1w bar
    close_1w = df_1w['close'].values
    close_1w_shifted = np.roll(close_1w, 1)
    close_1w_shifted[0] = np.nan
    ema_50_1w = pd.Series(close_1w_shifted).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: 20-period EMA of volume on 1d timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Camarilla levels (R3, S3) from prior completed 1d bar
    # Note: For 1d timeframe, we need to use prior completed 1d bar, so we shift by 1
    high_1d = np.roll(high, 1)
    low_1d = np.roll(low, 1)
    close_1d = np.roll(close, 1)
    high_1d[0] = np.nan
    low_1d[0] = np.nan
    close_1d[0] = np.nan
    
    camarilla_range = high_1d - low_1d
    r3 = close_1d + (1.1 * camarilla_range / 2)
    s3 = close_1d - (1.1 * camarilla_range / 2)
    
    # Note: r3 and s3 are already for prior completed 1d bar due to the roll above
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 + price above 1w EMA50 + volume spike
            if close[i] > r3[i] and close[i] > ema_50_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 + price below 1w EMA50 + volume spike
            elif close[i] < s3[i] and close[i] < ema_50_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint of Camarilla levels OR price crosses below 1w EMA50
            midpoint = (r3[i] + s3[i]) / 2
            if not np.isnan(midpoint) and (close[i] < midpoint or close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint of Camarilla levels OR price crosses above 1w EMA50
            midpoint = (r3[i] + s3[i]) / 2
            if not np.isnan(midpoint) and (close[i] > midpoint or close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals