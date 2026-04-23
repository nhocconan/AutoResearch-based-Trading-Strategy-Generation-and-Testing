#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian breakout (20-period) with 4h EMA50 trend filter and volume confirmation.
Long when price breaks above 20-bar high AND close > 4h EMA50 AND volume > 1.5x average.
Short when price breaks below 20-bar low AND close < 4h EMA50 AND volume > 1.5x average.
Exit on opposite Donchian break or trend reversal.
Uses 4h for signal direction, 1h only for entry timing to minimize trades and fee drag.
Designed for 1h timeframe targeting 60-150 total trades over 4 years (15-37/year) with session filter (08-20 UTC).
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
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Load 4h data for EMA50 trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) on 1h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_4h_aligned[i]
        donch_high = high_roll[i]
        donch_low = low_roll[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND price > 4h EMA50 (uptrend) AND volume spike
            if (price > donch_high and price > ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: price breaks below Donchian low AND price < 4h EMA50 (downtrend) AND volume spike
            elif (price < donch_low and price < ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.20
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian low OR trend reversal
                if (price < donch_low or price < ema50_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian high OR trend reversal
                if (price > donch_high or price > ema50_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Donchian20_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0