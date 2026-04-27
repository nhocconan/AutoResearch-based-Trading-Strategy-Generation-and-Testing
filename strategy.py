#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation.
# In bull markets: buy breakouts above 20-day high when weekly trend is up
# In bear markets: sell breakdowns below 20-day low when weekly trend is down
# Volume > 1.5x 20-day average confirms institutional participation.
# Designed for ~15-25 trades/year per symbol to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend filter: EMA(34) on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Bullish breakout: price closes above 20-day high
        if close[i] > highest_high[i]:
            # Only go long if weekly trend is up and volume confirms
            if close[i] > ema34_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
        
        # Bearish breakdown: price closes below 20-day low
        elif close[i] < lowest_low[i]:
            # Only go short if weekly trend is down and volume confirms
            if close[i] < ema34_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        # Exit conditions: reverse position on opposite signal or trail stop
        elif position == 1 and close[i] < lowest_low[i]:
            # Long position: exit if price breaks below 20-day low
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > highest_high[i]:
            # Short position: exit if price breaks above 20-day high
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "Daily_Donchian20_1wEMA34_VolumeFilter"
timeframe = "1d"
leverage = 1.0