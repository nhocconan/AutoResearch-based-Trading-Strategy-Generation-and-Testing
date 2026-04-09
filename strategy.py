#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout + 1d volume confirmation
# Camarilla levels calculated from prior 1d OHLC: H4/L4 = breakout levels
# In strong volume (>1.5x 20-period average): breakout continuation
# In weak volume: fade at H3/L3 levels
# Works in both bull/bear: adapts to volume context
# Position size 0.25 to limit drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_1d_camarilla_vol_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels from prior day OHLC
    # Camarilla: H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    #            H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        # Use prior day's OHLC
        rng = high_1d[i-1] - low_1d[i-1]
        camarilla_h4[i] = close_1d[i-1] + 1.5 * rng
        camarilla_l4[i] = close_1d[i-1] - 1.5 * rng
        camarilla_h3[i] = close_1d[i-1] + 1.125 * rng
        camarilla_l3[i] = close_1d[i-1] - 1.125 * rng
    
    # Align Camarilla levels to 6h timeframe
    h4_6h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_6h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_6h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_6h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_20 = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        vol_ma_20[i] = np.mean(vol_1d[i-19:i+1])
    vol_ma_20_6h = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h4_6h[i]) or np.isnan(l4_6h[i]) or 
            np.isnan(h3_6h[i]) or np.isnan(l3_6h[i]) or
            np.isnan(vol_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma_20_6h[i] if vol_ma_20_6h[i] > 0 else 1.0
        
        if position == 1:  # Long position
            # Exit conditions
            if close[i] <= h3_6h[i]:  # Exit at H3 level
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            if close[i] >= l3_6h[i]:  # Exit at L3 level
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic based on volume context
            if vol_ratio > 1.5:  # Strong volume - breakout continuation
                # Go long on break above H4
                if close[i] > h4_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Go short on break below L4
                elif close[i] < l4_6h[i]:
                    position = -1
                    signals[i] = -0.25
            else:  # Weak volume - fade at extremes
                # Go long at L3 support
                if close[i] < l3_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Go short at H3 resistance
                elif close[i] > h3_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals