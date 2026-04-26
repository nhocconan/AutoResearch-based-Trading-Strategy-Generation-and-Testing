#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Bounce_1wTrend_VolumeFilter
Hypothesis: On daily timeframe, price tends to bounce from Camarilla pivot support/resistance levels (S1/R1) when aligned with weekly trend and confirmed by volume spikes. 
Long when price touches S1 in weekly uptrend with volume > 1.8x 20-day MA. Short when price touches R1 in weekly downtrend with volume > 1.8x 20-day MA.
Uses discrete position sizing (0.25) to minimize fee churn. Weekly trend filter reduces counter-trend trades. Volume confirmation ensures institutional participation.
Target: 7-25 trades/year (30-100 total over 4 years) by requiring confluence of pivot touch, weekly trend alignment, and volume spike.
Works in both bull and bear markets by following the weekly trend direction for entries.
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
    
    # Get weekly data for trend filter (primary HTF as per experiment)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla S1 and R1 from prior daily OHLC (avoid look-ahead)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use prior day's OHLC for current day's levels
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev[0] = np.nan
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    
    # Camarilla S1 and R1 levels
    camarilla_range = high_1d_prev - low_1d_prev
    s1 = close_1d_prev - camarilla_range * 1.1 / 12  # Support level
    r1 = close_1d_prev + camarilla_range * 1.1 / 12  # Resistance level
    
    # Align Camarilla levels to daily timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Weekly EMA20 trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    uptrend_1w = close > ema_20_1w_aligned  # Price above weekly EMA20 = uptrend
    downtrend_1w = close < ema_20_1w_aligned  # Price below weekly EMA20 = downtrend
    
    # Volume confirmation: volume > 1.8x 20-day moving average (tight threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for weekly EMA + 20 for volume MA + 1 for daily shift)
    start_idx = 41
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price touches S1 (support) in weekly uptrend with volume spike
            if (low[i] <= s1_aligned[i] and  # Touch or penetrate support
                uptrend_1w[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches R1 (resistance) in weekly downtrend with volume spike
            elif (high[i] >= r1_aligned[i] and  # Touch or penetrate resistance
                  downtrend_1w[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes above R1 (resistance break) OR weekly trend turns down
            if (close[i] > r1_aligned[i] or not uptrend_1w[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes below S1 (support break) OR weekly trend turns up
            if (close[i] < s1_aligned[i] or not downtrend_1w[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_Pivot_Bounce_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0