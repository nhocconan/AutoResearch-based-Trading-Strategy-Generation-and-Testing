#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Breakout_VolumeTrend
Hypothesis: 12h price breaks above/below Camarilla R1/S1 levels with volume spike and 1d EMA34 trend confirmation.
Designed for 12h timeframe to reduce trade frequency and minimize fee drag while capturing significant breakouts.
Uses 1d EMA34 as trend filter to avoid counter-trend trades. Works in both bull and bear markets by following trend.
Target: 15-25 trades/year to stay within optimal range for 12h timeframe.
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
    
    # Camarilla levels from previous day (1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels for each 12h bar using previous day's data
    # Camarilla formulas: 
    # R4 = close + 1.5*(high-low)
    # R3 = close + 1.1*(high-low)
    # R2 = close + 0.6*(high-low)
    # R1 = close + 0.382*(high-low)
    # S1 = close - 0.382*(high-low)
    # S2 = close - 0.6*(high-low)
    # S3 = close - 1.1*(high-low)
    # S4 = close - 1.5*(high-low)
    hl_range_1d = high_1d - low_1d
    camarilla_r1 = close_1d + 0.382 * hl_range_1d
    camarilla_s1 = close_1d - 0.382 * hl_range_1d
    
    # Align Camarilla levels to 12h timeframe (using previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike: >1.8x 30-period average (higher threshold for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Trend filter: 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 35  # Warmup for EMA and Camarilla calculations
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        ema34 = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and uptrend
            if price > r1_level and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and downtrend
            elif price < s1_level and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below S1 OR trend turns down
            if price < s1_level:
                signals[i] = 0.0
                position = 0
            elif price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above R1 OR trend turns up
            if price > r1_level:
                signals[i] = 0.0
                position = 0
            elif price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_VolumeTrend"
timeframe = "12h"
leverage = 1.0