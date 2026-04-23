#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian channel breakout (20) with 1w EMA34 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND price > 1w EMA34 (uptrend) AND volume > 1.5x average.
Short when price breaks below Donchian lower band AND price < 1w EMA34 (downtrend) AND volume > 1.5x average.
Exit when price reverts to Donchian middle band or trend reverses (price crosses 1w EMA34).
Uses 1d timeframe for lower trade frequency (target: 30-100 trades over 4 years) to minimize fee drag.
1w EMA34 provides stable trend filter. Volume confirmation ensures high-conviction breakouts.
Works in both bull and bear markets: trend filter prevents counter-trend entries, volume confirms conviction.
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels (20-period) for 1d
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Load 1w data for EMA34 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 1d timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume average (20-period) on 1d timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_val = donchian_upper_aligned[i]
        lower_val = donchian_lower_aligned[i]
        middle_val = donchian_middle_aligned[i]
        ema34_val = ema34_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper band AND price > 1w EMA34 (uptrend) AND volume confirmation
            if (price > upper_val and price > ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND price < 1w EMA34 (downtrend) AND volume confirmation
            elif (price < lower_val and price < ema34_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to middle band OR price breaks below 1w EMA34 (trend reversal)
                if price <= middle_val or price < ema34_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to middle band OR price breaks above 1w EMA34 (trend reversal)
                if price >= middle_val or price > ema34_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0