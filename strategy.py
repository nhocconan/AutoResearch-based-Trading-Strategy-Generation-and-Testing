#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
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
    
    # Get 1h data for daily trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    close_1h = df_1h['close'].values
    ema_34_1h = pd.Series(close_1h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_34_1h)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    # R3 = Close + (High - Low) * 1.1 / 4
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    # S3 = Close - (High - Low) * 1.1 / 4
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    # R4 = Close + (High - Low) * 1.1 / 2
    r4_1d = close_1d + (range_1d * 1.1 / 2)
    # S4 = Close - (High - Low) * 1.1 / 2
    s4_1d = close_1d - (range_1d * 1.1 / 2)
    
    # Align daily Camarilla levels to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1h_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or
            np.isnan(s4_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + above daily EMA34 + volume confirmation
            if (close[i] > r3_1d_aligned[i] and 
                close[i] > ema_34_1h_aligned[i] and
                vol_ratio[i] > 1.8):
                # Avoid extreme extension beyond R4
                if close[i] <= r4_1d_aligned[i] * 1.02:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below S3 + below daily EMA34 + volume confirmation
            elif (close[i] < s3_1d_aligned[i] and 
                  close[i] < ema_34_1h_aligned[i] and
                  vol_ratio[i] > 1.8):
                # Avoid extreme extension beyond S4
                if close[i] >= s4_1d_aligned[i] * 0.98:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR below daily EMA34
            if close[i] < s3_1d_aligned[i] or close[i] < ema_34_1h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 OR above daily EMA34
            if close[i] > r3_1d_aligned[i] or close[i] > ema_34_1h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals