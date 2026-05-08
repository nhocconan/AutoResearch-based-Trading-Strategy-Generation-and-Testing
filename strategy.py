#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h EMA50 for trend filter ===
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # === Donchian(20) on 4h ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === Volume filter: current volume > 1.5 * 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback)  # warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price above Donchian upper band + uptrend + volume
            long_cond = (close[i] > highest_high[i] and 
                        close[i] > ema50_12h_aligned[i] and
                        volume[i] > vol_ma20[i] * 1.5)
            
            # Short breakdown: price below Donchian lower band + downtrend + volume
            short_cond = (close[i] < lowest_low[i] and 
                         close[i] < ema50_12h_aligned[i] and
                         volume[i] > vol_ma20[i] * 1.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian middle or trend reversal
            mid = (highest_high[i] + lowest_low[i]) / 2.0
            exit_cond = (close[i] < mid or close[i] < ema50_12h_aligned[i])
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Donchian middle or trend reversal
            mid = (highest_high[i] + lowest_low[i]) / 2.0
            exit_cond = (close[i] > mid or close[i] > ema50_12h_aligned[i])
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Donchian breakout with 12h EMA50 trend filter and volume confirmation.
# Enters long when price breaks above 20-period high with uptrend (above 12h EMA50) and high volume.
# Enters short when price breaks below 20-period low with downtrend (below 12h EMA50) and high volume.
# Exits when price crosses the Donchian middle or trend reverses.
# Designed to capture trends in both bull and bear markets with infrequent trades (target: 20-50/year).
# Uses discrete sizing (0.25) to minimize fee churn. Works on BTC/ETH via institutional price action.