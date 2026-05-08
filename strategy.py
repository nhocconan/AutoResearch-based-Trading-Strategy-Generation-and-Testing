#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with volume confirmation and 1d EMA trend filter.
# Uses Camarilla pivot levels (R3, S3) from 1d data for breakout entries.
# Long when price breaks above R3 with volume surge and 1d EMA up.
# Short when price breaks below S3 with volume surge and 1d EMA down.
# Exit when price crosses 12h EMA(20) or reverses pivot level.
# Works in both bull (breakouts with trend) and bear (breakouts against trend filtered by EMA).
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "12h_Camarilla_R3S3_Breakout_Volume_1dEMA"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # We use R3 and S3 for breakouts
    camarilla_R3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_S3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # 1d EMA(34) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_up = ema_34_1d[1:] > ema_34_1d[:-1]
    ema_34_1d_up = np.concatenate([[False], ema_34_1d_up])
    
    # Volume spike detection: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * vol_ma_20)
    
    # Align 1d Camarilla levels and EMA trend to 12h
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    ema_34_1d_up_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_up.astype(float))
    
    # 12h EMA(20) for exit
    close_series = pd.Series(close)
    ema_20_12h = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for 1d EMA(34)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(ema_34_1d_up_aligned[i]) or np.isnan(ema_20_12h[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 with volume spike and 1d EMA up
            if (close[i] > camarilla_R3_aligned[i] and 
                volume_spike[i] and 
                ema_34_1d_up_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 with volume spike and 1d EMA down
            elif (close[i] < camarilla_S3_aligned[i] and 
                  volume_spike[i] and 
                  not ema_34_1d_up_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 12h EMA(20) or re-enters below R3
            if close[i] < ema_20_12h[i] or close[i] < camarilla_R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 12h EMA(20) or re-enters above S3
            if close[i] > ema_20_12h[i] or close[i] > camarilla_S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals