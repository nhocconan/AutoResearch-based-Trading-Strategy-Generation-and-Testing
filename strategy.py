#!/usr/bin/env python3
"""
12h_Pivot_MeanReversion_1dTrend_VolumeFilter
Hypothesis: Trade mean reversion at daily Camarilla H3/L3 levels with 1d EMA34 trend filter and volume confirmation. 
In uptrend, buy L3 bounce; in downtrend, sell H3 rejection. Uses 12h timeframe for lower trade frequency.
Target: 15-30 trades/year, works in bull/bear by aligning with daily trend.
"""

name = "12h_Pivot_MeanReversion_1dTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0

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
    
    # === Daily OHLC for Camarilla Pivots ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values
    
    # Camarilla H3/L3 (mean reversion levels)
    camarilla_h3 = pc + (ph - pl) * 1.1 / 4
    camarilla_l3 = pc - (ph - pl) * 1.1 / 4
    
    # Align to 12h timeframe
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # === Daily Trend Filter (EMA34) ===
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Filter (1.5x 20-period EMA on 12h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers daily calculations)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or np.isnan(ema34_12h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price touches L3 in uptrend with volume (mean reversion long)
            if (low[i] <= l3_12h[i] and 
                close[i] > ema34_12h[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches H3 in downtrend with volume (mean reversion short)
            elif (high[i] >= h3_12h[i] and 
                  close[i] < ema34_12h[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches midpoint (mean reversion complete) or breaks H3 (trend resumption)
            if (close[i] >= pc[i] or close[i] >= h3_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price reaches midpoint or breaks L3 (trend resumption)
            if (close[i] <= pc[i] or close[i] <= l3_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals