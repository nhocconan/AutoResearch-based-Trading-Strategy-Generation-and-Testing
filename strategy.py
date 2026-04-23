#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian(20) breakout with 4h EMA(50) trend filter and volume confirmation.
Long when price breaks above 1h Donchian upper AND 4h EMA(50) upward AND volume > 1.5x average.
Short when price breaks below 1h Donchian lower AND 4h EMA(50) downward AND volume > 1.5x average.
Exit on opposite Donchian break or EMA trend reversal.
Uses 4h for signal direction, 1h only for entry timing precision. Session filter 08-20 UTC.
Target: 15-37 trades/year (60-150 over 4 years) to minimize fee drag. Works in bull/bear by only taking breakouts in direction of 4h trend.
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
    
    # Precompute session hours (08-20 UTC) - prices.index is DatetimeIndex
    hours = prices.index.hour
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA(50) for trend direction
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h EMA(50) slope (trend strength)
    ema_slope_4h = np.zeros_like(ema_50_4h_aligned)
    ema_slope_4h[1:] = (ema_50_4h_aligned[1:] - ema_50_4h_aligned[:-1]) / ema_50_4h_aligned[:-1]
    
    # Calculate 1h Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) on 1h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema_50_4h_aligned[i]
        ema_slope = ema_slope_4h[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND 4h EMA upward AND volume spike
            if (price > upper and ema_slope > 0 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Donchian lower AND 4h EMA downward AND volume spike
            elif (price < lower and ema_slope < 0 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian lower OR 4h EMA slope turns down
                if (price < lower or ema_slope < 0):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian upper OR 4h EMA slope turns up
                if (price > upper or ema_slope > 0):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Donchian20_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0