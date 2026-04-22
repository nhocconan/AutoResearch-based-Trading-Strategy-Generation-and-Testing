#!/usr/bin/env python3
"""
Hypothesis: 6h/1d Volume-Weighted Average Price (VWAP) with 1-day volume filter.
Long when price > VWAP and 1-day volume > 50-period average volume.
Short when price < VWAP and 1-day volume > 50-period average volume.
Exit when price crosses VWAP or 1-day volume drops below average.
VWAP provides intraday mean reversion; volume filter ensures institutional participation.
Works in both bull and bear markets by following institutional volume while using VWAP for entry timing.
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
    
    # Load 1-day data for volume filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(volume_1d).rolling(window=50, min_periods=50).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # VWAP calculation for 6h period
    typical_price = (high + low + close) / 3.0
    vwap_numerator = typical_price * volume
    vwap_denominator = volume
    
    # Cumulative VWAP (resets daily)
    vwap = np.full(n, np.nan)
    cum_num = 0.0
    cum_den = 0.0
    
    for i in range(n):
        # Reset at start of each day (00:00 UTC)
        if i > 0 and prices['open_time'].iloc[i].date() != prices['open_time'].iloc[i-1].date():
            cum_num = 0.0
            cum_den = 0.0
        
        cum_num += vwap_numerator[i]
        cum_den += vwap_denominator[i]
        
        if cum_den > 0:
            vwap[i] = cum_num / cum_den
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(vwap[i]) or np.isnan(avg_vol_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above VWAP and 1-day volume above average
            if close[i] > vwap[i] and volume_1d[i] > avg_vol_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below VWAP and 1-day volume above average
            elif close[i] < vwap[i] and volume_1d[i] > avg_vol_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below VWAP
                if close[i] < vwap[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above VWAP
                if close[i] > vwap[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_VWAP_1dVolume_Filter"
timeframe = "6h"
leverage = 1.0