# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_1dTrend_Volume
Hypothesis: On the 6h timeframe, breakouts at the extreme Camarilla levels (R4/S4) 
represent institutional breakouts with follow-through. We use 1d trend filter (EMA50) 
to ensure we trade in the direction of the daily trend, and volume confirmation 
to avoid false breakouts. This strategy targets 50-150 total trades over 4 years 
(12-37/year) by focusing on high-probability breakout scenarios.
Works in both bull and bear markets by aligning with the daily trend direction.
"""

name = "6h_Camarilla_R4_S4_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous day
    # R4 = close + (high - low) * 1.12 / 2
    # S4 = close - (high - low) * 1.12 / 2
    camarilla_r4 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.12 / 2
    camarilla_s4 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.12 / 2
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4.values)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4.values)
    
    # Volume confirmation (20-period MA on 6h = ~5 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d EMA50 (50), Camarilla (needs 1d), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + price breaks above R4 + volume
            if uptrend and close[i] > camarilla_r4_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below S4 + volume
            elif downtrend and close[i] < camarilla_s4_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below R4
            if not uptrend or close[i] < camarilla_r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters above S4
            if not downtrend or close[i] > camarilla_s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals