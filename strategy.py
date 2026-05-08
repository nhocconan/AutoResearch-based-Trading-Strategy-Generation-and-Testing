#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + 1w trend filter
# Long when price breaks above 12h Donchian upper band, volume > 1.5x 12-period average, 1w EMA(34) rising
# Short when price breaks below 12h Donchian lower band, volume > 1.5x 12-period average, 1w EMA(34) falling
# Donchian provides clear breakout signals, volume confirms institutional interest, weekly trend filters counter-trend moves
# Designed for 12h timeframe to capture multi-day trends with minimal trades (target: 15-35/year)

name = "12h_Donchian20_VolumeSpike_1wTrend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily and weekly data once for filters
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # 12-period Donchian channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_up = high_roll.values
    donchian_low = low_roll.values
    
    # 12-period average volume for spike detection
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    # Weekly EMA(34) for trend filter
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for Donchian and volume calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_up[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > Donchian upper, volume spike, weekly uptrend
            if close[i] > donchian_up[i] and volume[i] > 1.5 * vol_ma[i] and ema34_1w_aligned[i] > ema34_1w_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Enter short: price < Donchian lower, volume spike, weekly downtrend
            elif close[i] < donchian_low[i] and volume[i] > 1.5 * vol_ma[i] and ema34_1w_aligned[i] < ema34_1w_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Donchian lower or weekly trend turns down
            if close[i] < donchian_low[i] or ema34_1w_aligned[i] < ema34_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Donchian upper or weekly trend turns up
            if close[i] > donchian_up[i] or ema34_1w_aligned[i] > ema34_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals