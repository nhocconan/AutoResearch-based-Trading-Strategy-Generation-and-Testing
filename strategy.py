#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot (R1/S1) breakout with volume confirmation and 1d EMA34 trend filter.
- Camarilla levels from daily OHLC provide institutional support/resistance
- Breakout above R1 or below S1 captures momentum with high probability
- Volume spike (>1.5x 20-period average) confirms institutional participation
- 1d EMA34 filter ensures trades align with higher timeframe trend
- Target: 20-40 trades/year to avoid fee drag
- Uses discrete position sizing (0.25) to minimize churn
"""

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
    
    # Get daily data for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1_1d = close_1d + range_1d * 1.1 / 12
    s1_1d = close_1d - range_1d * 1.1 / 12
    
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema34_1d[i] = (close_1d[i] * 2 / 35) + (ema34_1d[i-1] * 33 / 35)
    
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = max(35, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(ema34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 + volume spike + price > EMA34 (uptrend)
            if (close[i] > r1_1d_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 + volume spike + price < EMA34 (downtrend)
            elif (close[i] < s1_1d_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below S1 OR price < EMA34 (trend reversal)
            if (close[i] < s1_1d_aligned[i] or 
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 OR price > EMA34 (trend reversal)
            if (close[i] > r1_1d_aligned[i] or 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_VolumeSpike_EMA34_Trend_v1"
timeframe = "4h"
leverage = 1.0