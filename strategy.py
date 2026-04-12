#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_Volume_Confirmation_v1
Hypothesis: On 4h timeframe, buy when price breaks above Camarilla R4 resistance level (calculated from prior 1d OHLC) with volume confirmation, sell when price breaks below Camarilla S4 support level with volume confirmation. Uses Camarilla pivot levels for institutional reference points, volume confirmation for breakout strength, and avoids overtrading by requiring significant breaks of outer Camarilla levels (R4/S4). Works in bull markets via R4 breakouts and in bear markets via S4 breakdowns. Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Pivot_Breakout_Volume_Confirmation_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average (20 period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior 1d OHLC
    # Camarilla: R4 = Close + 1.5 * (High - Low), S4 = Close - 1.5 * (High - Low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_s4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align to 4h timeframe (values from prior 1d bar available at 4h bar open)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate pivot point for exit
    pivot_point = (high_1d + low_1d + close_1d) / 3
    pivot_point_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(pivot_point_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Camarilla breakout conditions
        breakout_r4 = close[i] > camarilla_r4_aligned[i]  # Break above R4
        breakout_s4 = close[i] < camarilla_s4_aligned[i]  # Break below S4
        
        # Volume confirmation: current volume > 1.5x average
        volume_spike = volume[i] > vol_ma[i] * 1.5
        
        # Entry conditions
        long_entry = breakout_r4 and volume_spike
        short_entry = breakout_s4 and volume_spike
        
        # Exit conditions: price returns to Camarilla pivot point (midpoint)
        long_exit = close[i] < pivot_point_aligned[i]
        short_exit = close[i] > pivot_point_aligned[i]
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals