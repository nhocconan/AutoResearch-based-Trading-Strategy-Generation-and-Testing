#!/usr/bin/env python3
"""
4h_12h_Camarilla_Pivot_Breakout_Volume_Filter
Hypothesis: Uses 12-hour Camarilla pivot levels for breakout entries with volume confirmation and 4-hour momentum filter.
Designed for low trade frequency (<30/year) with high-probability setups in both bull and bear markets.
Uses 12h timeframe for structural levels to reduce noise and false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_Pivot_Breakout_Volume_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Momentum filter: 4h RSI(14) to avoid choppy markets
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from 12h data
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Camarilla levels: R4 = close + 1.5*(high-low), S4 = close - 1.5*(high-low)
    camarilla_r4 = close_12h + 1.5 * (high_12h - low_12h)
    camarilla_s4 = close_12h - 1.5 * (high_12h - low_12h)
    
    # Align Camarilla levels to 4h timeframe (wait for 12h close)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        # Momentum filter: RSI between 30 and 70 to avoid extremes
        momentum_filter = (rsi[i] >= 30) and (rsi[i] <= 70)
        
        # Breakout conditions using 12h Camarilla levels
        breakout_up = close[i] > camarilla_r4_aligned[i]   # Break above R4
        breakdown_down = close[i] < camarilla_s4_aligned[i] # Break below S4
        
        # Entry conditions: require volume and momentum filters
        long_entry = breakout_up and volume_filter and momentum_filter
        short_entry = breakdown_down and volume_filter and momentum_filter
        
        # Exit conditions: return to opposite Camarilla level
        long_exit = close[i] < camarilla_s4_aligned[i]
        short_exit = close[i] > camarilla_r4_aligned[i]
        
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