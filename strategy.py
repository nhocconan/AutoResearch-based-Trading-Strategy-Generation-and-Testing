#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian breakout with 1-day ATR volatility filter and volume confirmation.
Long when price breaks above 20-period Donchian high, ATR(14) > 1.5x its 50-period average (high volatility regime), and volume > 1.5x average.
Short when price breaks below 20-period Donchian low, ATR(14) > 1.5x its 50-period average, and volume > 1.5x average.
Exit when price crosses the 20-period Donchian midline or volatility drops.
Designed for low trade frequency (~15-25/year) to capture volatile breakouts while avoiding chop.
Works in both bull and bear markets by requiring high volatility conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for ATR - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1-day ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Load 12-hour data for Donchian channels - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Align HTF indicators to lower timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma)
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_12h, donch_mid)
    
    # Volume average (20-period) on lower timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(atr_aligned[i]) or np.isnan(atr_ma_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        atr_val = atr_aligned[i]
        atr_ma_val = atr_ma_aligned[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        donch_mid_val = donch_mid_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        close_val = close[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high, high volatility, volume confirmation
            if (close_val > donch_high_val and 
                atr_val > 1.5 * atr_ma_val and 
                vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, high volatility, volume confirmation
            elif (close_val < donch_low_val and 
                  atr_val > 1.5 * atr_ma_val and 
                  vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below Donchian midline OR volatility drops
                if close_val < donch_mid_val or atr_val < atr_ma_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above Donchian midline OR volatility drops
                if close_val > donch_mid_val or atr_val < atr_ma_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_ATR_Volume_Breakout"
timeframe = "12h"
leverage = 1.0