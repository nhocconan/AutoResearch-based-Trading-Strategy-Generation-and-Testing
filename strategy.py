#!/usr/bin/env python3
"""
4h_12h_Camarilla_R3_S3_Breakout_TrendFilter_Volume
Hypothesis: Uses Camarilla pivot levels from 12h timeframe (R3/S3) for breakout entries on 4h chart,
confirmed by 12h EMA50 trend and volume spikes. Designed for low trade frequency by requiring confluence of
price breaking key 12h pivot levels, trend alignment, and volume confirmation. Works in bull and bear
markets by following intermediate-term trend from 12h timeframe.
"""

name = "4h_12h_Camarilla_R3_S3_Breakout_TrendFilter_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 12h OHLCV for Camarilla Pivot Levels ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate pivot points using previous 12h's OHLC
    prev_high_12h = df_12h['high'].values
    prev_low_12h = df_12h['low'].values
    prev_close_12h = df_12h['close'].values
    
    pivot_12h = (prev_high_12h + prev_low_12h + prev_close_12h) / 3.0
    range_val_12h = prev_high_12h - prev_low_12h
    
    # Camarilla levels (R3 and S3)
    R3_12h = pivot_12h + (range_val_12h * 1.1 / 4)
    S3_12h = pivot_12h - (range_val_12h * 1.1 / 4)
    
    # Align to 4h timeframe
    R3_4h = align_htf_to_ltf(prices, df_12h, R3_12h)
    S3_4h = align_htf_to_ltf(prices, df_12h, S3_12h)
    
    # --- 12h EMA50 Trend Filter ---
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # --- Volume Spike Detection (20-period average on 4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA50 and pivot calculation)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN (first few bars)
        if (np.isnan(R3_4h[i]) or np.isnan(S3_4h[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(vol_ratio[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.8
        
        if position == 0:
            # Long: price breaks above R3 with volume, above EMA50
            if (close[i] > R3_4h[i] and 
                volume_spike and 
                close[i] > ema_50_4h[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume, below EMA50
            elif (close[i] < S3_4h[i] and 
                  volume_spike and 
                  close[i] < ema_50_4h[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite breakout or loss of momentum
            if position == 1:
                # Exit long: price breaks below S3 (reversal signal)
                if close[i] < S3_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above R3 (reversal signal)
                if close[i] > R3_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals