#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's Camarilla levels (shifted by 1 to avoid look-ahead)
    R3 = high_1d + (high_1d - low_1d) * 1.1 / 6
    S3 = low_1d - (high_1d - low_1d) * 1.1 / 6
    R4 = high_1d + (high_1d - low_1d) * 1.1 / 2
    S4 = low_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Shift by 1 to use only previous day's data
    R3 = np.roll(R3, 1)
    S3 = np.roll(S3, 1)
    R4 = np.roll(R4, 1)
    S4 = np.roll(S4, 1)
    R3[0] = S3[0] = R4[0] = S4[0] = np.nan
    
    # Align Camarilla levels to 4h timeframe
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    R4_4h = align_htf_to_ltf(prices, df_1d, R4)
    S4_4h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Daily EMA34 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume spike detection (4h timeframe)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_4h[i]) or np.isnan(S3_4h[i]) or np.isnan(R4_4h[i]) or 
            np.isnan(S4_4h[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]  # Require strong volume spike
        
        if position == 0:
            # Long: Price breaks above R3 with daily uptrend and volume spike
            if close[i] > R3_4h[i] and close[i] > ema_1d_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with daily downtrend and volume spike
            elif close[i] < S3_4h[i] and close[i] < ema_1d_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below S3 or trend turns down
            if close[i] < S3_4h[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above R3 or trend turns up
            if close[i] > R3_4h[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals