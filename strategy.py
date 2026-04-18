#!/usr/bin/env python3
"""
12h_Pivot_R1S1_Breakout_With_Volume_and_TrendFilter
Hypothesis: Breakout at daily Camarilla R1/S1 levels on 12h timeframe with volume confirmation and 1d EMA trend filter.
Designed for 12-37 trades/year to avoid fee drag while capturing breakout moves in trending markets.
Works in both bull and bear markets by aligning with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels."""
    typical = (high + low + close) / 3
    range_val = high - low
    R1 = close + range_val * 1.1 / 12
    S1 = close - range_val * 1.1 / 12
    return R1, S1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla R1/S1 on 1d
    R1, S1 = calculate_camarilla(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    R1_1d = align_htf_to_ltf(prices, df_1d, R1)
    S1_1d = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: >1.5x 20-period average (12h bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 20)  # Warmup for EMA and volume
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R1_1d[i]) or
            np.isnan(S1_1d[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        R1_val = R1_1d[i]
        S1_val = S1_1d[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and uptrend
            if not np.isnan(R1_val) and price > R1_val and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and downtrend
            elif not np.isnan(S1_val) and price < S1_val and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price breaks below S1 OR trend turns down
            if not np.isnan(S1_val) and price < S1_val:
                signals[i] = 0.0
                position = 0
            elif price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price breaks above R1 OR trend turns up
            if not np.isnan(R1_val) and price > R1_val:
                signals[i] = 0.0
                position = 0
            elif price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Pivot_R1S1_Breakout_With_Volume_and_TrendFilter"
timeframe = "12h"
leverage = 1.0