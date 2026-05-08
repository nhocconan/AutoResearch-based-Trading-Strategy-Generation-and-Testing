#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with daily trend filter and volume confirmation
# We go long when price breaks above the 20-period high with daily EMA(34) uptrend and volume spike.
# We go short when price breaks below the 20-period low with daily EMA(34) downtrend and volume spike.
# Uses 12h timeframe to target 12-37 trades/year, avoiding excessive frequency.
# Donchian channels provide clear breakout levels that work in trending markets.
# Daily trend filter ensures we trade with the higher timeframe momentum.
# Volume spike confirms institutional participation in the breakout.

name = "12h_Donchian20_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    daily_close = df_1d['close'].values
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Donchian(20) channels on 12h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        ema34_1d_val = ema34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high + daily uptrend + volume spike
            if close[i] > donchian_high_val and close[i] > ema34_1d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + daily downtrend + volume spike
            elif close[i] < donchian_low_val and close[i] < ema34_1d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR daily trend turns down
            if close[i] < donchian_low_val or close[i] < ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high OR daily trend turns up
            if close[i] > donchian_high_val or close[i] > ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals