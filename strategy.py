#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above upper Donchian(20) AND price > 12h EMA50 (uptrend) AND volume > 1.8x average.
Short when price breaks below lower Donchian(20) AND price < 12h EMA50 (downtrend) AND volume > 1.8x average.
Exit when price crosses 12h EMA50 (trend reversal) or reaches opposite Donchian band.
Uses 4h timeframe to target ~30-60 trades/year, balancing signal quality and fee drag.
Works in both bull and bear markets by requiring trend confirmation via 12h EMA50 for breakout entries.
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
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 for 12h trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Donchian channels (20-period) on 4h timeframe
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_12h_aligned[i]
        upper_val = donchian_upper[i]
        lower_val = donchian_lower[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND price > 12h EMA50 (uptrend) AND volume spike
            if (price > upper_val and price > ema50_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND price < 12h EMA50 (downtrend) AND volume spike
            elif (price < lower_val and price < ema50_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below 12h EMA50 (trend reversal) OR reaches lower Donchian
                if price < ema50_val or price < lower_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above 12h EMA50 (trend reversal) OR reaches upper Donchian
                if price > ema50_val or price > upper_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_12hEMA50_Volume_Breakout"
timeframe = "4h"
leverage = 1.0