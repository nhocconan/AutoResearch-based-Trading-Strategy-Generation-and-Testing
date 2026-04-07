#!/usr/bin/env python3
"""
12h_donchian_breakout_1w_trend_volume_v1
Hypothesis: On 12h timeframe, enter long when price breaks above 20-bar Donchian high with above-average volume and weekly trend up (price > weekly SMA50), enter short when price breaks below 20-bar Donchian low with above-average volume and weekly trend down (price < weekly SMA50). Exit when price crosses the 20-bar Donchian midline (mean of high/low channel). Uses weekly trend filter to avoid counter-trend trades. Designed for 15-30 trades/year to minimize fee decay while capturing major trend moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    if len(high) < 20:
        return np.zeros(n)
    
    # Donchian high and low
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0  # midline for exit
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly SMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i]) or np.isnan(sma50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian midline
            if close[i] < donch_mid[i] and close[i-1] >= donch_mid[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian midline
            if close[i] > donch_mid[i] and close[i-1] <= donch_mid[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price breaks above Donchian high with weekly uptrend
                if close[i] > donch_high[i] and close[i-1] <= donch_high[i-1] and close[i] > sma50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low with weekly downtrend
                elif close[i] < donch_low[i] and close[i-1] >= donch_low[i-1] and close[i] < sma50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals