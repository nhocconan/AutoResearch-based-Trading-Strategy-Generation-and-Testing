#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout (20) with volume confirmation and weekly trend filter
# Long when price breaks above Donchian high (20) with volume > 1.5x average, weekly uptrend
# Short when price breaks below Donchian low (20) with volume > 1.5x average, weekly downtrend
# Uses volume surge to confirm breakout strength, weekly trend for higher timeframe alignment
# Designed to work in both bull and bear markets by capturing strong directional moves
# Targets 30-100 total trades over 4 years (7-25/year) to minimize fee drag

name = "1d_Donchian20_Volume_WeeklyTrend"
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
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Donchian Channels (20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high with volume confirmation, weekly uptrend
            if close[i] > donchian_high[i] and volume[i] > 1.5 * vol_avg[i] and ema34_1w_aligned[i] > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low with volume confirmation, weekly downtrend
            elif close[i] < donchian_low[i] and volume[i] > 1.5 * vol_avg[i] and ema34_1w_aligned[i] < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or weekly trend turns down
            if close[i] < donchian_low[i] or ema34_1w_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or weekly trend turns up
            if close[i] > donchian_high[i] or ema34_1w_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals