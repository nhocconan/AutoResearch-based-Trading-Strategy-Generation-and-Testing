#!/usr/bin/env python3
# Strategy: 4h_1d_Pivot_R1S1_Breakout_Volume_TrendFilter_v1
# Hypothesis: Breakout above daily Pivot R1 or below S1 on 4h with volume confirmation (1.5x 20-bar MA) and 1d EMA50 trend filter.
# Uses 4h bars for entries, filtered by 1d trend to avoid counter-trend trades. Volume confirms institutional interest.
# Targets 20-40 trades/year to minimize fee drag and work in both bull/bear markets via trend alignment.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for HTF analysis (trend, pivot levels)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Daily Pivot levels (standard)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    R1 = pivot_1d + (range_1d * 1.0 / 2)  # R1 = P + (H-L)*0.5
    S1 = pivot_1d - (range_1d * 1.0 / 2)  # S1 = P - (H-L)*0.5
    
    # Align 1d indicators to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Load 4h data for entry timing, volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Volume spike detection (20-period on 4h)
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        
        if position == 0:
            # Long: price breaks above R1, above 1d EMA50 (uptrend), with volume confirmation
            if (price > R1_aligned[i] and 
                price > ema50_1d_aligned[i] and 
                vol > 1.5 * vol_ma_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, below 1d EMA50 (downtrend), with volume confirmation
            elif (price < S1_aligned[i] and 
                  price < ema50_1d_aligned[i] and 
                  vol > 1.5 * vol_ma_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1
            if price < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1
            if price > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Pivot_R1S1_Breakout_Volume_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0