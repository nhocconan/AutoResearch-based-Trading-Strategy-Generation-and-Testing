#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with 1-day ATR-based volatility filter and volume confirmation.
# Donchian(20) captures breakouts from price channels, which work well in trending markets.
# ATR filter ensures we only trade when volatility is expanding (avoiding choppy markets).
# Volume confirmation ensures breakouts have participation. Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year).
# Works in both bull and bear markets by filtering for volatility expansion and using symmetric long/short logic.

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
    
    # Get daily data for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for ATR calculation
        return np.zeros(n)
    
    # Calculate daily ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR using Wilder's smoothing (equivalent to RMA)
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate ATR ratio: current ATR / 20-period average ATR
    # Values > 1 indicate expanding volatility
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = np.where(atr_ma > 0, atr / atr_ma, 1.0)
    
    # Volatility filter: ATR ratio > 1.2 (expanding volatility)
    vol_filter = atr_ratio > 1.2
    
    # Align volatility filter to 4h timeframe
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    
    # Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vol_filter_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when volatility is expanding
        vol_expanding = vol_filter_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # Break above 20-period high
        breakout_down = close[i] < lowest_low[i-1]  # Break below 20-period low
        
        # Entry conditions with volume confirmation
        long_entry = vol_expanding and breakout_up and volume_filter[i]
        short_entry = vol_expanding and breakout_down and volume_filter[i]
        
        # Exit conditions: when price returns to the opposite Donchian boundary
        # Exit long when price touches or crosses below the lower band
        long_exit = position == 1 and close[i] <= lowest_low[i]
        # Exit short when price touches or crosses above the upper band
        short_exit = position == -1 and close[i] >= highest_high[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_DonchianBreakout_1dATR_VolFilter_Volume"
timeframe = "4h"
leverage = 1.0