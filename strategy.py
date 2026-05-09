#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(50) trend filter and volume confirmation.
# Donchian breakout provides clear entry/exit levels, 12h EMA ensures trend alignment,
# volume confirmation filters false breakouts. Designed to work in both bull (follow 12h uptrend)
# and bear (follow 12h downtrend) markets by taking long/short based on EMA direction.
# Target: 20-40 trades/year to avoid fee drag.
name = "4h_Donchian20_12hEMA50_Volume_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need sufficient lookback for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian upper + above 12h EMA + volume
            if (price > highest_high[i] and price > ema_50_12h_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian lower + below 12h EMA + volume
            elif (price < lowest_low[i] and price < ema_50_12h_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian lower or below 12h EMA
            if price < lowest_low[i] or price < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian upper or above 12h EMA
            if price > highest_high[i] or price > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals