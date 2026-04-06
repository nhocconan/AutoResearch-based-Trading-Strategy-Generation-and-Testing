#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above 20-day high AND weekly close > weekly EMA20 AND volume > 1.5x average
# Short when price breaks below 20-day low AND weekly close < weekly EMA20 AND volume > 1.5x average
# Exit when price returns to 10-day midpoint OR weekly trend reverses
# Uses daily timeframe for entries with weekly trend filter to reduce false signals
# Designed to capture medium-term trends while avoiding whipsaws in choppy markets

name = "1d_donchian20_weekly_trend_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend filter - EMA20 on weekly closes
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    
    # Weekly close aligned
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
    
    # Volume confirmation: volume > 1.5x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(weekly_ema20_aligned[i]) or np.isnan(weekly_close_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            # Exit: price returns to 10-day midpoint OR weekly trend turns bearish
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] <= midpoint or weekly_close_aligned[i] < weekly_ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to 10-day midpoint OR weekly trend turns bullish
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] >= midpoint or weekly_close_aligned[i] > weekly_ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with weekly trend confirmation + volume
            # Long: price breaks above 20-day high AND weekly close > weekly EMA20 AND volume confirmation
            if (close[i] > highest_high[i] and 
                weekly_close_aligned[i] > weekly_ema20_aligned[i] and 
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND weekly close < weekly EMA20 AND volume confirmation
            elif (close[i] < lowest_low[i] and 
                  weekly_close_aligned[i] < weekly_ema20_aligned[i] and 
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
    
    return signals