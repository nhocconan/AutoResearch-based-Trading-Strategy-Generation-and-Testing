#!/usr/bin/env python3
"""
12h_donchian_breakout_1w_trend_volume_v1
Hypothesis: On 12h timeframe, use weekly Donchian breakouts with weekly EMA trend filter and volume confirmation to capture major trends. This strategy targets 50-150 total trades over 4 years (12-37/year) by requiring confluence of price breaking weekly Donchian channels, alignment with weekly EMA trend, and above-average volume. Works in both bull and bear markets by only trading in the direction of the weekly trend, avoiding counter-trend whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly Donchian channels and EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Donchian channels (20-period)
    donch_high_1w = np.full_like(high_1w, np.nan)
    donch_low_1w = np.full_like(low_1w, np.nan)
    
    for i in range(20, len(df_1w)):
        donch_high_1w[i] = np.max(high_1w[i-20:i])
        donch_low_1w[i] = np.min(low_1w[i-20:i])
    
    # Weekly EMA (20-period) for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly indicators to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high_1w)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low_1w)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate volume moving average for confirmation (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly Donchian low or reverses against trend
            if close[i] < donch_low_aligned[i] or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above weekly Donchian high or reverses against trend
            if close[i] > donch_high_aligned[i] or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price breaks above weekly Donchian high with upward trend
                if close[i] > donch_high_aligned[i] and close[i] > ema_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below weekly Donchian low with downward trend
                elif close[i] < donch_low_aligned[i] and close[i] < ema_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals