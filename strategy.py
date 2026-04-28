#!/usr/bin/env python3
# Hypothesis: 1-day Donchian channel breakout with weekly ATR volatility filter and volume confirmation.
# Uses 20-day Donchian channels for trend following, filtered by weekly ATR volatility regime
# (only trade when volatility is above average) and volume confirmation to avoid false breakouts.
# Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year).
# Works in bull markets by catching breakouts and in bear markets by avoiding low-volatility chop.

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
    
    # Get weekly data for ATR volatility filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for ATR calculation
        return np.zeros(n)
    
    # Calculate weekly ATR (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Wilder's smoothing for ATR
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate ATR ratio: current ATR / 50-period average ATR
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma
    atr_ratio[np.isnan(atr_ratio)] = 1.0  # Handle NaN from insufficient data
    
    # Align ATR ratio to daily timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio)
    
    # Daily Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when volatility is above average (ATR ratio > 1.0)
        high_volatility = atr_ratio_aligned[i] > 1.0
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_roll[i-1]  # Break above previous period's high
        breakout_down = close[i] < low_roll[i-1]  # Break below previous period's low
        
        # Entry conditions with volume confirmation
        long_entry = high_volatility and breakout_up and volume_filter[i]
        short_entry = high_volatility and breakout_down and volume_filter[i]
        
        # Exit conditions: opposite breakout or volatility drops
        long_exit = breakout_down or (not high_volatility)
        short_exit = breakout_up or (not high_volatility)
        
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

name = "1d_DonchianBreakout_1wATR_VolumeFilter"
timeframe = "1d"
leverage = 1.0