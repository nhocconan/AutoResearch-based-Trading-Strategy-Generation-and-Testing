#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 12-hour trend filter and volume confirmation.
Long when price breaks above Donchian(20) high, 12-hour EMA(50) is rising, and volume > 1.5x average.
Short when price breaks below Donchian(20) low, 12-hour EMA(50) is falling, and volume > 1.5x average.
Exit when price returns to Donchian midpoint or trend reverses.
Designed for low trade frequency (~20-40/year) to capture breakouts in trending markets.
Works in both bull and bear markets by requiring 12-hour trend confirmation.
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
    
    # Calculate Donchian channels (20-period) on 4h
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Load 12-hour data for EMA trend - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # EMA(50) on 12h
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) on 12h
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to lower timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Volume average (20-period) on lower timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        donch_mid_val = donch_mid[i]
        ema_50_val = ema_50_aligned[i]
        vol_ma_12h_val = vol_ma_12h_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        close_val = close[i]
        
        if position == 0:
            # Long: Break above Donchian high, rising EMA, volume confirmation
            if (close_val > donch_high_val and
                ema_50_val > ema_50_aligned[i-1] and  # EMA rising
                vol_current > 1.5 * vol_ma_12h_val):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low, falling EMA, volume confirmation
            elif (close_val < donch_low_val and
                  ema_50_val < ema_50_aligned[i-1] and  # EMA falling
                  vol_current > 1.5 * vol_ma_12h_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to midpoint OR EMA starts falling
                if close_val <= donch_mid_val or ema_50_val < ema_50_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to midpoint OR EMA starts rising
                if close_val >= donch_mid_val or ema_50_val > ema_50_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0