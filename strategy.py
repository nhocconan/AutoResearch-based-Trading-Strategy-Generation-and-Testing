#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wTrendFilter_VolumeSpike
Hypothesis: Trade daily Camarilla R1/S1 breakouts with weekly EMA34 trend filter and volume spike confirmation. 
Uses 1d primary timeframe to reduce trade frequency and fee drag. Weekly trend filter ensures alignment with major market direction.
Volume spike confirms breakout strength. Discrete sizing (0.25) to limit fee changes.
Target: 15-25 trades/year per symbol (~60-100 total over 4 years) to minimize fee impact.
Designed to work in both bull (breakouts with trend) and bear (mean reversion at extremes) markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for HTF trend filter and Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from previous weekly bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla levels: R1/S1 (using close of previous weekly bar)
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_1w + 1.1 * (high_1w - low_1w) / 12
    camarilla_s1 = close_1w - 1.1 * (high_1w - low_1w) / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine weekly HTF trend (bullish = price above EMA34)
        htf_1w_bullish = close[i] > ema_34_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above R1 + weekly uptrend + volume spike
            long_setup = (close[i] > camarilla_r1_aligned[i]) and htf_1w_bullish and volume_spike[i]
            
            # Short setup: price breaks below S1 + weekly downtrend + volume spike
            short_setup = (close[i] < camarilla_s1_aligned[i]) and htf_1w_bearish and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches S1 (stop) OR weekly trend turns bearish
            if (close[i] <= camarilla_s1_aligned[i]) or (not htf_1w_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches R1 (stop) OR weekly trend turns bullish
            if (close[i] >= camarilla_r1_aligned[i]) or (htf_1w_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wTrendFilter_VolumeSpike"
timeframe = "1d"
leverage = 1.0