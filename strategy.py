#!/usr/bin/env python3
"""
12h_1d_1w_Volume_Weighted_Pivot_Breakout
Hypothesis: Use daily volume-weighted pivot (VWAP) as a dynamic support/resistance level. 
Breakouts above the pivot with volume expansion indicate strong institutional participation, 
while breakdowns below the pivot with volume expansion indicate distribution. 
Weekly trend filter ensures trades align with higher timeframe momentum. 
Target: 20-30 trades/year to minimize fee drag.
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
    
    # Get daily data for VWAP calculation (volume-weighted average price)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily VWAP: typical price * volume cumulative / volume cumulative
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_values = vwap.values
    
    # Align daily VWAP to 12h
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_values)
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    # Weekly EMA20 for trend filter
    ema_20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(vwap_aligned[i]) or np.isnan(ema_20_weekly_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: breakout above VWAP with volume expansion and price above weekly EMA20
        long_condition = (close[i] > vwap_aligned[i]) and volume_expansion[i] and (close[i] > ema_20_weekly_aligned[i])
        
        # Short: breakdown below VWAP with volume expansion and price below weekly EMA20
        short_condition = (close[i] < vwap_aligned[i]) and volume_expansion[i] and (close[i] < ema_20_weekly_aligned[i])
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "12h_1d_1w_Volume_Weighted_Pivot_Breakout"
timeframe = "12h"
leverage = 1.0