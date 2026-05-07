#!/usr/bin/env python3
name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HT"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Camarilla: range = high - low
    # R4 = close + range * 1.1/2
    # R3 = close + range * 1.1/4
    # S3 = close - range * 1.1/4
    # S4 = close - range * 1.1/2
    range_1d = df_1d['high'] - df_1d['low']
    close_1d = df_1d['close']
    R3 = close_1d + range_1d * 1.1 / 4
    S3 = close_1d - range_1d * 1.1 / 4
    R4 = close_1d + range_1d * 1.1 / 2
    S4 = close_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 6h
    R3_6h = align_htf_to_ltf(prices, df_1d, R3.values)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3.values)
    R4_6h = align_htf_to_ltf(prices, df_1d, R4.values)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4.values)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (2x 20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough history
    
    for i in range(start_idx, n):
        if (np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or np.isnan(R4_6h[i]) or 
            np.isnan(S4_6h[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: Break above R3 with volume in 1d uptrend
            if close[i] > R3_6h[i] and vol_condition and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with volume in 1d downtrend
            elif close[i] < S3_6h[i] and vol_condition and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price re-enters below R3 or trend changes
            if close[i] < R3_6h[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price re-enters above S3 or trend changes
            if close[i] > S3_6h[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend and volume spike
# - Camarilla R3/S3 levels act as support/resistance from prior day
# - Breakout above R3 (with volume) in 1d uptrend = long
# - Breakout below S3 (with volume) in 1d downtrend = short
# - Volume confirmation (2x 20-period MA) reduces false breakouts
# - Exit when price re-enters R3/S3 zone or trend changes
# - Position size 0.25 targets ~50-150 total trades over 4 years
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Uses actual 1d Camarilla calculations (not resampled) via mtf_data
# - Novel for 6h timeframe: combines intraday breakout with daily structure
# - Aims for 60-120 total trades to stay within 6h limits (15-30/year)