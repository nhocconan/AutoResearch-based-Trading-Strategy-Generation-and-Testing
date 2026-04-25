#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrendFilter_VolumeSpike_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 12h with 1d EMA34 trend filter and volume spike confirmation. Camarilla levels provide precise intraday support/resistance. In bull markets (price > 1d EMA34), buy breakouts above R1; in bear markets (price < 1d EMA34), sell breakdowns below S1. Volume spike confirms institutional interest. Designed for 12-25 trades/year to minimize fee drag while capturing trends in both bull and bear regimes.
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
    
    # Get 1d data for HTF trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for previous 1d (using prior completed 1d bar)
    # Camarilla: based on previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values  # previous 1d high
    prev_low = df_1d['low'].shift(1).values    # previous 1d low
    prev_close = df_1d['close'].shift(1).values # previous 1d close
    
    # True range for Camarilla calculation
    tr = prev_high - prev_low
    
    # Camarilla levels (R1, S1) - inner levels for breakout trading
    r1 = prev_close + (tr * 1.1 / 12)
    s1 = prev_close - (tr * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe (no extra delay needed as they're based on completed 1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above R1 + 1d uptrend + volume spike
            long_setup = (close[i] > r1_aligned[i]) and htf_1d_bullish and volume_spike[i]
            
            # Short setup: price breaks below S1 + 1d downtrend + volume spike
            short_setup = (close[i] < s1_aligned[i]) and htf_1d_bearish and volume_spike[i]
            
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
            # Exit: price touches S1 (stop) OR 1d trend turns bearish
            if (close[i] <= s1_aligned[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches R1 (stop) OR 1d trend turns bullish
            if (close[i] >= r1_aligned[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrendFilter_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0