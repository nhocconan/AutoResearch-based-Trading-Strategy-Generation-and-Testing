#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R1S1_Breakout_Volume
Hypothesis: Camarilla pivot levels (R1, S1) from 1d timeframe act as support/resistance on 12h.
Breakout above R1 or below S1 with volume confirmation and trend filter (1d EMA34) captures
strong moves in both bull and bear markets. Uses 1d timeframe for pivots and trend filter.
Target: 15-25 trades/year to minimize fee drag while capturing strong trend continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for pivot calculation and EMA34 trend filter (loaded once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    hl_range = high_1d - low_1d
    r1 = close_1d + 1.1 * hl_range / 12
    s1 = close_1d - 1.1 * hl_range / 12
    
    # Align pivot levels to 12h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 1)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema34 = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: breakout above R1 with volume spike in uptrend
            if (price > r1_val and
                vol_spike and
                price > ema34):
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S1 with volume spike in downtrend
            elif (price < s1_val and
                  vol_spike and
                  price < ema34):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below S1 (reversal) or trend reverses
            if close[i] < s1_val or price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above R1 (reversal) or trend reverses
            if close[i] > r1_val or price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_Pivot_R1S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0