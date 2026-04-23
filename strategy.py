#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike.
Long when price breaks above 20-day high AND price > 1w EMA50 (uptrend) AND volume > 2.0x 20-day average.
Short when price breaks below 20-day low AND price < 1w EMA50 (downtrend) AND volume > 2.0x 20-day average.
Exit when price reverts to 10-day EMA or trend reverses (price crosses 1w EMA50).
Uses 1d timeframe to target 15-30 trades/year, avoiding fee drag while capturing strong breakouts.
1w EMA50 provides smooth trend filter to reduce whipsaw. Volume confirmation ensures high-conviction breakouts.
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
    
    # Load 1d data for Donchian channels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channels (20-period) on 1d
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-day EMA for exit
    ema10_1d = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Align 1d indicators to lower timeframe (prices timeframe)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema10_1d_aligned = align_htf_to_ltf(prices, df_1d, ema10_1d)
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 for 1w trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (20-period) on 1d timeframe
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(ema10_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        ema50_val = ema50_1w_aligned[i]
        ema10_val = ema10_1d_aligned[i]
        vol_ma_val = vol_ma_1d_aligned[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above 20-day high AND price > 1w EMA50 (uptrend) AND volume spike
            if (price > donch_high and price > ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND price < 1w EMA50 (downtrend) AND volume spike
            elif (price < donch_low and price < ema50_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to 10-day EMA OR price breaks below 1w EMA50 (trend reversal)
                if price <= ema10_val or price < ema50_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to 10-day EMA OR price breaks above 1w EMA50 (trend reversal)
                if price >= ema10_val or price > ema50_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA50_Volume_Breakout"
timeframe = "1d"
leverage = 1.0