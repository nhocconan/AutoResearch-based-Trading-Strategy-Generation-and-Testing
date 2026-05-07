#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with weekly EMA34 trend filter and volume confirmation.
# Long when price breaks above 20-day high AND weekly EMA34 rising AND volume > 1.5x 20-day average.
# Short when price breaks below 20-day low AND weekly EMA34 falling AND volume > 1.5x 20-day average.
# Exit when price crosses back inside Donchian channel (above 20-day low for long, below 20-day high for short).
# This strategy targets volatility expansion phases with higher timeframe trend alignment to capture momentum moves
# while avoiding choppy markets. Weekly EMA34 filter ensures we trade with the primary trend on higher timeframe.
# Volume confirmation ensures institutional participation and reduces false breakouts.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the weekly trend direction.

name = "1d_DonchianBreakout_WeeklyEMA34_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-day high/low)
    donch_length = 20
    highest_high = pd.Series(high).rolling(window=donch_length, min_periods=donch_length).max().values
    lowest_low = pd.Series(low).rolling(window=donch_length, min_periods=donch_length).min().values
    
    # Weekly EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Weekly EMA34 direction
    ema34_rising = np.zeros_like(ema34_1w_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1w_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1w_aligned[1:] > ema34_1w_aligned[:-1]
    ema34_falling[1:] = ema34_1w_aligned[1:] < ema34_1w_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donch_length, 35)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(ema34_rising[i]) or np.isnan(ema34_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above 20-day high, weekly EMA34 rising, volume filter
            long_cond = (close[i] > highest_high[i]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: price breaks below 20-day low, weekly EMA34 falling, volume filter
            short_cond = (close[i] < lowest_low[i]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back inside Donchian channel (above 20-day low)
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back inside Donchian channel (below 20-day high)
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals