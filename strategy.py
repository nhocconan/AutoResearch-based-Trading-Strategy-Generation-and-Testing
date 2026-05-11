#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hTrend_VolumeS
Hypothesis: Uses Camarilla pivot levels from daily data for entry/exit, with 12h EMA trend filter and volume confirmation.
Long when price breaks above R1 with volume and price > 12h EMA50; short when breaks below S1 with volume and price < 12h EMA50.
Exit on opposite Camarilla level touch. Designed for moderate trade frequency with trend alignment to work in both bull and bear markets.
"""
name = "4h_Camarilla_R1S1_Breakout_12hTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivots and 12h data for trend
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 2 or len(df_12h) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Camarilla Pivot Levels (R1, S1) ---
    # Classic Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_range = high_1d - low_1d
    r1 = close_1d + camarilla_range * 1.1 / 12
    s1 = close_1d - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 4h (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # --- 12h EMA50 Trend Filter ---
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (ensure 12h EMA has enough data)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above R1 with volume and above 12h EMA50
            if (close[i] > r1_aligned[i] and 
                volume_spike and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and below 12h EMA50
            elif (close[i] < s1_aligned[i] and 
                  volume_spike and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: touch opposite Camarilla level
            if position == 1:
                # Exit long: price touches or goes below S1
                if close[i] <= s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches or goes above R1
                if close[i] >= r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals