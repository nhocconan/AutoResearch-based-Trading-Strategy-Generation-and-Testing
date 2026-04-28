#!/usr/bin/env python3
# Hypothesis: 6h momentum breakout using 1-day Donchian channels with volume and volatility confirmation.
# Donchian breakouts capture strong trends, while 1-day timeframe filters out noise.
# Volume surge (2x average) confirms institutional participation.
# Volatility filter (ATR ratio > 1.2) avoids whipsaws in low-volatility ranging markets.
# Works in bull markets via breakout continuation and bear markets via breakdowns.
# Target: 50-150 total trades over 4 years (12-37/year) with selective entries.

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
    
    # Get daily data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: highest high over 20 days
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over 20 days
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily ATR (14-period) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR calculation
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 50-period average ATR (volatility regime filter)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.where(atr_ma > 0, atr / atr_ma, 1.0)
    
    # Align indicators to 6h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume filter: volume > 2.0 x 20-period average on 6s timeframe
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when volatility is above average
        vol_filter = atr_ratio_aligned[i] > 1.2
        
        # Breakout conditions
        long_breakout = close[i] > upper_20_aligned[i]
        short_breakout = close[i] < lower_20_aligned[i]
        
        # Entry conditions with volume and volatility confirmation
        long_entry = long_breakout and volume_filter[i] and vol_filter
        short_entry = short_breakout and volume_filter[i] and vol_filter
        
        # Exit conditions: opposite breakout or volatility collapse
        long_exit = (short_breakout) or (atr_ratio_aligned[i] < 0.8)
        short_exit = (long_breakout) or (atr_ratio_aligned[i] < 0.8)
        
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

name = "6h_DonchianBreakout_1dVolumeVolatilityFilter"
timeframe = "6h"
leverage = 1.0