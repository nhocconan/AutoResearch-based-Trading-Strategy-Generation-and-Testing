#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above 1d Donchian upper band AND price > 1w EMA50 (uptrend) AND volume > 2.0x average.
Short when price breaks below 1d Donchian lower band AND price < 1w EMA50 (downtrend) AND volume > 2.0x average.
Exit when price reverts to 1d Donchian midpoint or trend reverses (price crosses 1w EMA50).
Uses 1d timeframe to minimize trades and fee drag. Donchian provides clear structure, 1w EMA50 smooth trend filter.
Volume spike ensures high-conviction breakouts. Target: 50-80 trades over 4 years (12-20/year) to stay within proven working range.
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
    
    # Calculate 1d Donchian(20) - ONCE before loop
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1d timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, prices, donchian_upper)  # same timeframe
    donchian_lower_aligned = align_htf_to_ltf(prices, prices, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, prices, donchian_mid)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (20-period) on 1d timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_val = donchian_upper_aligned[i]
        lower_val = donchian_lower_aligned[i]
        mid_val = donchian_mid_aligned[i]
        ema50_val = ema50_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper AND price > 1w EMA50 (uptrend) AND volume spike
            if (price > upper_val and price > ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower AND price < 1w EMA50 (downtrend) AND volume spike
            elif (price < lower_val and price < ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to 1d Donchian midpoint OR price breaks below 1w EMA50 (trend reversal)
                if price <= mid_val or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to 1d Donchian midpoint OR price breaks above 1w EMA50 (trend reversal)
                if price >= mid_val or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0