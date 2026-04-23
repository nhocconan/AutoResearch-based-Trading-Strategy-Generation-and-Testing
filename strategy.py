#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h trend filter (price > 12h EMA50 for long, price < 12h EMA50 for short) and volume confirmation (>1.5x average).
Uses 6h timeframe to reduce trade frequency vs lower timeframes, targeting 50-150 total trades over 4 years.
12h EMA50 provides medium-term trend filter to avoid counter-trend trades.
Volume confirmation ensures breakouts have conviction.
Works in both bull and bear markets: trend filter prevents whipsaws in ranging markets, volume confirms breakout strength.
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
    
    # Calculate 6h Donchian channels (20-period) - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Donchian channels on 6h (based on previous 20 periods)
    upper_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_6h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_6h, lower_20)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma_primary[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_val = upper_20_aligned[i]
        lower_val = lower_20_aligned[i]
        ema50_val = ema50_12h_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above 6h Donchian upper AND price > 12h EMA50 (uptrend) AND volume confirmation
            if (price > upper_val and price > ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 6h Donchian lower AND price < 12h EMA50 (downtrend) AND volume confirmation
            elif (price < lower_val and price < ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below 6h Donchian lower OR price breaks below 12h EMA50 (trend reversal)
                if price < lower_val or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above 6h Donchian upper OR price breaks above 12h EMA50 (trend reversal)
                if price > upper_val or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0