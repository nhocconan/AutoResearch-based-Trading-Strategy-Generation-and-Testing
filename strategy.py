#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Filtered_v2
Hypothesis: Refined version with stricter conditions - requires volume spike AND 
price closing beyond the pivot level (not just touching) to reduce false breaks. 
Maintains daily trend filter for alignment. Targets 12-30 trades/year to stay 
well within fee limits while capturing only high-probability breaks.
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
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    # Using R4/S4 for stronger breakout signals
    camarilla_range = (high_1d - low_1d) * 1.1 / 2
    r4_1d = close_1d + camarilla_range
    s4_1d = close_1d - camarilla_range
    
    # Align R4/S4 to 12h timeframe (use previous day's levels)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Daily trend filter: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 2.5 * 20-period average (stricter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and volume MA
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        r4 = r4_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        ema_trend = ema50_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price closes above R4 with uptrend and volume spike
            if close[i] > r4 and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: price closes below S4 with downtrend and volume spike
            elif close[i] < s4 and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below S4 or trend turns down
            if close[i] < s4 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above R4 or trend turns up
            if close[i] > r4 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume_Filtered_v2"
timeframe = "12h"
leverage = 1.0