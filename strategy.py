#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrendFilter_v1
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter.
- Uses 12h timeframe for low trade frequency (target: 50-150 total trades over 4 years)
- Camarilla pivot levels (R1, S1) calculated from prior 1d candle
- Long when price breaks above R1 AND 1d trend up (close > EMA34)
- Short when price breaks below S1 AND 1d trend down (close < EMA34)
- Exit when price returns to Camarilla H3/L3 levels or opposite breakout
- Designed for 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
- Works in bull/bear markets by trading with the 1d trend and using Camarilla for precise entries
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load daily data ONCE before loop for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla pivot levels from prior 1d candle
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    # R2 = close + 0.7*(high-low), R1 = close + 0.5*(high-low)
    # PP = (high + low + close)/3
    # S1 = close - 0.5*(high-low), S2 = close - 0.7*(high-low), 
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels
    range_1d = high_1d - low_1d
    r1 = close_1d + 0.5 * range_1d
    s1 = close_1d - 0.5 * range_1d
    # Also calculate H3/L3 for exits (optional, using same 0.5 factor)
    h3 = close_1d + 0.5 * range_1d  # Same as R1 in Camarilla
    l3 = close_1d - 0.5 * range_1d  # Same as S1 in Camarilla
    
    # Align Camarilla levels to current timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 1 for daily data alignment)
    start_idx = 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions
        price_above_r1 = close[i] > r1_aligned[i]
        price_below_s1 = close[i] < s1_aligned[i]
        
        # Daily trend filter
        daily_uptrend = close[i] > ema34_1d_aligned[i]
        daily_downtrend = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 AND daily uptrend
            if price_above_r1 and daily_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND daily downtrend
            elif price_below_s1 and daily_downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below S1 OR daily trend turns down
            if price_below_s1 or not daily_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above R1 OR daily trend turns up
            if price_above_r1 or not daily_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrendFilter_v1"
timeframe = "12h"
leverage = 1.0