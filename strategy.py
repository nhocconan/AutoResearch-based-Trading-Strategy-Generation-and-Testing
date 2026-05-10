#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_1dVolume
Hypothesis: On 12h timeframe, use Camarilla pivot levels (R1, S1) from prior 1d for breakout entries, filtered by 1w EMA34 trend and 1d volume surge. This captures breakouts with trend and volume confirmation, suitable for both bull and bear markets via trend filter. Targets 15-25 trades/year to minimize fee drag.
"""

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_1dVolume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot calculation (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values  # previous day high
    low_prev = df_1d['low'].shift(1).values    # previous day low
    close_prev = df_1d['close'].shift(1).values # previous day close
    
    # Camarilla R1 and S1
    R1 = close_prev + (high_prev - low_prev) * 1.1 / 12
    S1 = close_prev - (high_prev - low_prev) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (no extra delay needed as levels are based on closed daily bar)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get 1d data for volume filter (20-period average)
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # 12h data for price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Camarilla (1d lag), 1w EMA34 (34), volume MA (20)
    start_idx = 34  # covers all warmup periods
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or
            np.isnan(ema34_1w_aligned[i]) or
            np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 1w EMA34
        uptrend_1w = close[i] > ema34_1w_aligned[i]
        downtrend_1w = close[i] < ema34_1w_aligned[i]
        
        # Volume filter: current 12h volume > 1.5x 1d 20-period MA
        volume_filter = volume[i] > vol_ma20_1d_aligned[i] * 1.5
        
        # Session filter: 08-20 UTC (adjust for 12h bars)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long: price breaks above R1 in uptrend with volume
            if high[i] > R1_aligned[i] and uptrend_1w and volume_filter and in_session:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 in downtrend with volume
            elif low[i] < S1_aligned[i] and downtrend_1w and volume_filter and in_session:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or trend fails
            if low[i] < S1_aligned[i] or not uptrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 or trend fails
            if high[i] > R1_aligned[i] or not downtrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals