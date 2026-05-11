# 6h_Weekly_Pivot_Daily_Trend_Filter_v1
# Hypothesis: Uses weekly pivot points from 1w timeframe to determine market bias, and daily EMA34 for trend filtering.
# Enters long when price breaks above weekly R1 with volume confirmation and above daily EMA34.
# Enters short when price breaks below weekly S1 with volume confirmation and below daily EMA34.
# Uses volume spike (1.5x 20-period average) for confirmation. Designed for low trade frequency by requiring
# multiple confluence factors: weekly pivot breakout, volume confirmation, and trend alignment.
# Works in both bull and bear markets by following the trend filter (daily EMA34) which adapts to market conditions.

name = "6h_Weekly_Pivot_Daily_Trend_Filter_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly Pivot Points (standard calculation) ---
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align weekly levels to 6h
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # --- Daily Trend Filter (EMA34 on 1d close) ---
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_34_aligned[i])):
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
            # Long: price breaks above weekly R1 with volume, above daily EMA34
            if (close[i] > r1_1w_aligned[i] and 
                volume_spike and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume, below daily EMA34
            elif (close[i] < s1_1w_aligned[i] and 
                  volume_spike and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite breakout or loss of trend
            if position == 1:
                # Exit long: price breaks below weekly S1 or below daily EMA34
                if (close[i] < s1_1w_aligned[i] or 
                    close[i] < ema_34_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above weekly R1 or above daily EMA34
                if (close[i] > r1_1w_aligned[i] or 
                    close[i] > ema_34_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals