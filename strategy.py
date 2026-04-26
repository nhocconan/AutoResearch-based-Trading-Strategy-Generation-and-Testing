#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation to capture major moves while minimizing trades.
- Uses 12h timeframe targeting 50-150 total trades over 4 years (12-37/year)
- Camarilla R1/S1 from previous 1w bar for structure
- Long when price breaks above R1 with volume spike and 1w uptrend
- Short when price breaks below S1 with volume spike and 1w downtrend
- Volume spike filter ensures institutional participation
- Designed for low trade frequency to minimize fee drag while working in both bull and bear markets via trend filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Camarilla levels from previous 1w bar
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    camarilla_range = (high_1w - low_1w) * 1.1 / 12
    r1_1w = close_1w_arr + camarilla_range
    s1_1w = close_1w_arr - camarilla_range
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1w bar)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Calculate volume spike (20-period volume average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 34 for EMA)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions with volume confirmation and trend filter
        price_above_r1 = close[i] > r1_aligned[i]
        price_below_s1 = close[i] < s1_aligned[i]
        
        # 1w trend filter
        trend_up = close[i] > ema34_1w_aligned[i]
        trend_down = close[i] < ema34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 AND volume spike AND 1w uptrend
            if price_above_r1 and volume_spike[i] and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume spike AND 1w downtrend
            elif price_below_s1 and volume_spike[i] and trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below S1 OR 1w trend turns down
            if price_below_s1 or not trend_up:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above R1 OR 1w trend turns up
            if price_above_r1 or not trend_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0