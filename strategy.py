#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with daily trend filter and volume confirmation
# Go long when price breaks above 12h Donchian upper channel + daily EMA(34) uptrend + volume spike
# Go short when price breaks below 12h Donchian lower channel + daily EMA(34) downtrend + volume spike
# Uses 12h timeframe to target 12-37 trades/year, avoiding excessive frequency.
# Donchian channels provide clear breakout levels that work in trending markets.
# Daily trend filter ensures we trade with higher timeframe momentum.
# Volume spike confirms institutional participation in breakouts.

name = "12h_Donchian20_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate 12h Donchian channels (20-period)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        upper[i] = np.max(high[i - lookback + 1:i + 1])
        lower[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume spike: current volume > 2.0 * 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        upper_val = upper[i]
        lower_val = lower[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian upper + daily uptrend + volume spike
            if close[i] > upper_val and close[i] > ema34_1d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian lower + daily downtrend + volume spike
            elif close[i] < lower_val and close[i] < ema34_1d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower OR daily trend turns down
            if close[i] < lower_val or close[i] < ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian upper OR daily trend turns up
            if close[i] > upper_val or close[i] > ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals