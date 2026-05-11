#!/usr/bin/env python3
"""
1d_Weekly_Camarilla_R3_S3_Breakout_TrendFilter_Volume
Hypothesis: Uses weekly Camarilla pivot levels (R3/S3) for breakout entries on daily chart,
confirmed by weekly EMA20 trend and volume spikes. Designed for low trade frequency by requiring
confluence of price breaking key weekly pivot levels, trend alignment, and volume confirmation.
Works in bull and bear markets by following intermediate-term trend from weekly timeframe.
"""

name = "1d_Weekly_Camarilla_R3_S3_Breakout_TrendFilter_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly OHLCV for Camarilla Pivot Levels ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate pivot points using previous week's OHLC
    prev_high_1w = df_1w['high'].values
    prev_low_1w = df_1w['low'].values
    prev_close_1w = df_1w['close'].values
    
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    range_val_1w = prev_high_1w - prev_low_1w
    
    # Camarilla levels (R3 and S3)
    R3_1w = pivot_1w + (range_val_1w * 1.1 / 4)
    S3_1w = pivot_1w - (range_val_1w * 1.1 / 4)
    
    # Align to daily timeframe
    R3_1d = align_htf_to_ltf(prices, df_1w, R3_1w)
    S3_1d = align_htf_to_ltf(prices, df_1w, S3_1w)
    
    # --- Weekly EMA20 Trend Filter ---
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # --- Volume Spike Detection (20-period average on daily) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA20 and pivot calculation)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN (first few bars)
        if (np.isnan(R3_1d[i]) or np.isnan(S3_1d[i]) or 
            np.isnan(ema_20_1d[i]) or np.isnan(vol_ratio[i])):
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
            # Long: price breaks above R3 with volume, above EMA20
            if (close[i] > R3_1d[i] and 
                volume_spike and 
                close[i] > ema_20_1d[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume, below EMA20
            elif (close[i] < S3_1d[i] and 
                  volume_spike and 
                  close[i] < ema_20_1d[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite breakout or loss of momentum
            if position == 1:
                # Exit long: price breaks below S3 (reversal signal)
                if close[i] < S3_1d[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above R3 (reversal signal)
                if close[i] > R3_1d[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals