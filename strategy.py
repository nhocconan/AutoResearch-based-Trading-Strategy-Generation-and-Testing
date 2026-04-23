#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above 20-period 6h Donchian high AND price > 1d EMA50 (uptrend) AND volume > 2.0x average.
Short when price breaks below 20-period 6h Donchian low AND price < 1d EMA50 (downtrend) AND volume > 2.0x average.
Exit when price reverts to 6h Donchian midpoint or trend reverses (price crosses 1d EMA50).
Uses 6h timeframe for lower trade frequency to minimize fee drag while capturing multi-day trends.
1d EMA50 provides stable trend filter. Volume confirmation ensures high-conviction breakouts.
Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.
Target: 75-150 trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 6h data for Donchian channels - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate Donchian channels for 6h (20-period)
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 6h indicators to 6h timeframe (no alignment needed for same timeframe)
    # But we need to align 1d EMA50 to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_6h, donchian_mid)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1d_aligned[i]
        high_val = donchian_high_aligned[i]
        low_val = donchian_low_aligned[i]
        mid_val = donchian_mid_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND price > 1d EMA50 (uptrend) AND volume spike
            if (price > high_val and price > ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND price < 1d EMA50 (downtrend) AND volume spike
            elif (price < low_val and price < ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to Donchian midpoint OR price breaks below 1d EMA50 (trend reversal)
                if price <= mid_val or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to Donchian midpoint OR price breaks above 1d EMA50 (trend reversal)
                if price >= mid_val or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0