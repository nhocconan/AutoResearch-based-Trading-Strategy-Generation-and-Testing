#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
Long when price breaks above Donchian upper(20) AND price > 1w EMA34 (uptrend) AND volume > 1.3x average.
Short when price breaks below Donchian lower(20) AND price < 1w EMA34 (downtrend) AND volume > 1.3x average.
Exit when price crosses Donchian midpoint or trend reverses (price crosses 1w EMA34).
Designed for very low trade frequency (~15-25/year) to capture breakouts in trending markets while avoiding false signals in ranging conditions.
Works in both bull and bear markets by requiring trend confirmation via 1w EMA34 for breakout entries.
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
    
    # Load 1d data for Donchian channels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 1d timeframe
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Load 1w data for EMA34 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 for 1w trend filter
    ema34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF arrays to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34)
    
    # Volume average (20-period) on 1d timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        dh = donchian_high_aligned[i]
        dl = donchian_low_aligned[i]
        dm = donchian_mid_aligned[i]
        ema34_val = ema34_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper(20) AND price > 1w EMA34 (uptrend) AND volume spike
            if (price > dh and price > ema34_val and vol_current > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower(20) AND price < 1w EMA34 (downtrend) AND volume spike
            elif (price < dl and price < ema34_val and vol_current > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses Donchian midpoint OR price breaks below 1w EMA34 (trend reversal)
                if price < dm or price < ema34_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses Donchian midpoint OR price breaks above 1w EMA34 (trend reversal)
                if price > dm or price > ema34_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA34_Volume_Breakout"
timeframe = "1d"
leverage = 1.0