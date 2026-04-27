#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation.
# In bull markets: buy breakouts above 20-period high when 12h EMA50 is rising.
# In bear markets: sell breakdowns below 20-period low when 12h EMA50 is falling.
# Volume > 1.5x 20-period average confirms institutional participation.
# Designed for ~25-40 trades/year per symbol to avoid fee drag.

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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 50-period EMA on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Bullish breakout: price breaks above 20-period high
        if close[i] > highest_high[i]:
            # Only take long if 12h trend is up (EMA50 rising)
            if i > start_idx and ema50_12h_aligned[i] > ema50_12h_aligned[i-1]:
                if volume_filter[i]:
                    signals[i] = 0.30
                    position = 1
        
        # Bearish breakdown: price breaks below 20-period low
        elif close[i] < lowest_low[i]:
            # Only take short if 12h trend is down (EMA50 falling)
            if i > start_idx and ema50_12h_aligned[i] < ema50_12h_aligned[i-1]:
                if volume_filter[i]:
                    signals[i] = -0.30
                    position = -1
        
        # Exit conditions: reverse signal on opposite breakout
        elif position == 1 and close[i] < lowest_low[i]:
            signals[i] = -0.30  # Reverse to short
            position = -1
        elif position == -1 and close[i] > highest_high[i]:
            signals[i] = 0.30   # Reverse to long
            position = 1
        
        # Hold current position
        elif position == 1:
            signals[i] = 0.30
        elif position == -1:
            signals[i] = -0.30
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0