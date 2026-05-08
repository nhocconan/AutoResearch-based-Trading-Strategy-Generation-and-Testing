#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Confirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily high/low/close for Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, S3)
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # Resistance levels
    r3_1d = close_1d + range_1d * 1.1000 / 4  # R3 = C + (H-L)*1.1/4
    r4_1d = close_1d + range_1d * 1.1000 / 2  # R4 = C + (H-L)*1.1/2
    # Support levels
    s3_1d = close_1d - range_1d * 1.1000 / 4  # S3 = C - (H-L)*1.1/4
    s4_1d = close_1d - range_1d * 1.1000 / 2  # S4 = C - (H-L)*1.1/2
    
    # Align daily levels to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(r4_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(s4_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + above daily EMA34 + volume confirmation
            if (close[i] > r3_1d_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and
                vol_ratio[i] > 1.5):
                # Avoid extreme extension beyond R4
                if close[i] <= r4_1d_aligned[i] * 1.02:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below S3 + below daily EMA34 + volume confirmation
            elif (close[i] < s3_1d_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and
                  vol_ratio[i] > 1.5):
                # Avoid extreme extension beyond S4
                if close[i] >= s4_1d_aligned[i] * 0.98:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price below R3 OR below daily EMA34
            if close[i] < r3_1d_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above S3 OR above daily EMA34
            if close[i] > s3_1d_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals