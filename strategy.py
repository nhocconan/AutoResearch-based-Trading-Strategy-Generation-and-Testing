#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation (>1.8x 20-bar avg volume).
# Uses Donchian channel breakouts for momentum capture, 1w EMA50 for primary trend alignment (works in bull/bear via trend filter),
# and high volume threshold to reduce false signals. Designed for very low trade frequency (<50 total 1d trades over 4 years)
# to minimize fee drag while capturing strong trending moves. Weekly EMA ensures we only trade with the dominant weekly trend.

name = "1d_Donchian20_1wEMA50_VolumeBreakout_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough for 20-bar Donchian and 50-bar weekly EMA
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter (loaded ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period)
    lookback_donch = 20
    highest_high = pd.Series(high).rolling(window=lookback_donch, min_periods=lookback_donch).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=lookback_donch, min_periods=lookback_donch).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback_donch, min_periods=lookback_donch).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback_donch, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper, close > 1w EMA50, volume spike (>1.8x avg)
            if (high[i] > highest_high[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below Donchian lower, close < 1w EMA50, volume spike (>1.8x avg)
            elif (low[i] < lowest_low[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close if price breaks below Donchian lower or volume drops significantly
            if (low[i] < lowest_low[i]) or (volume[i] < 0.4 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Close if price breaks above Donchian upper or volume drops significantly
            if (high[i] > highest_high[i]) or (volume[i] < 0.4 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals