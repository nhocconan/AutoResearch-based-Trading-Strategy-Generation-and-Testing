#!/usr/bin/env python3
"""
1d_Weekly_Camarilla_R3_S3_Breakout_Trend_Volume
Hypothesis: Uses weekly Camarilla pivot levels (R3/S3) from 1w timeframe as key support/resistance.
On 1d timeframe, price breaking above R3 with volume confirmation and above weekly EMA50 trend triggers long.
Price breaking below S3 with volume confirmation and below weekly EMA50 trend triggers short.
Designed for low trade frequency by requiring weekly level breaks + volume + trend alignment.
Works in bull markets (breakouts continue) and bear markets (breakdowns continue) by following weekly trend.
"""

name = "1d_Weekly_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for Camarilla pivots and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly Camarilla Pivot Levels (R3, S3) ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot point
    pivot = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    r3 = close_1w + range_1w * 1.1 / 4.0
    s3 = close_1w - range_1w * 1.1 / 4.0
    
    # Align weekly levels to daily
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # --- Weekly EMA50 Trend Filter ---
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # --- Volume Spike Detection (20-day average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i])):
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
            # Long: price breaks above weekly R3 with volume, above weekly EMA50
            if (close[i] > r3_aligned[i] and 
                volume_spike and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S3 with volume, below weekly EMA50
            elif (close[i] < s3_aligned[i] and 
                  volume_spike and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite break or loss of trend
            if position == 1:
                # Exit long: price breaks below weekly S3 or below EMA50
                if (close[i] < s3_aligned[i] or 
                    close[i] < ema_50_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above weekly R3 or above EMA50
                if (close[i] > r3_aligned[i] or 
                    close[i] > ema_50_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals