#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h EMA(50) trend filter and volume confirmation
- Uses Donchian channel breakout (20-period high/low) for entry signals
- 12h EMA(50) defines trend direction (only long when price > EMA, short when price < EMA)
- Volume confirmation (> 1.5x 20-period average) filters low-momentum breakouts
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading with the 12h trend
- Volume spike requirement reduces false breakouts during low volatility
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
    
    # Calculate Donchian channel (20-period high/low)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        price_above_donchian_high = close[i] > highest_high[i]
        price_below_donchian_low = close[i] < lowest_low[i]
        
        # Trend filter: price > EMA for long, price < EMA for short
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, uptrend, volume spike
            long_signal = (price_above_donchian_high and 
                          uptrend and
                          volume[i] > 1.5 * vol_ma[i])
            
            # Short conditions: price breaks below Donchian low, downtrend, volume spike
            short_signal = (price_below_donchian_low and 
                           downtrend and
                           volume[i] > 1.5 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Donchian breakout or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below Donchian low or trend turns down
                if (price_below_donchian_low or 
                    not uptrend):  # Trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above Donchian high or trend turns up
                if (price_above_donchian_high or 
                    not downtrend):  # Trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0