# 1D_Wyckoff_Spring_1WeekTrend_Volume
# Hypothesis: Wyckoff Spring pattern on 1d (price drop below support then quick reversal with volume) filtered by 1w trend and volume spike captures accumulation in both bull and bear markets.
# Works in bull: buys dips in uptrend. Works in bear: catches true bottoms before rallies.
# Uses 1d for entry, 1w for trend filter (HTF). Low trade frequency expected (<25/year) due to strict pattern.

#!/usr/bin/env python3
name = "1D_Wyckoff_Spring_1WeekTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 20-period EMA on weekly close for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1-week EMA20 to daily timeframe (waits for weekly close)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Need at least 20 days for lookback
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if weekly EMA not ready
        if np.isnan(ema20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1-week trend
        uptrend = close[i] > ema20_1w_aligned[i]
        downtrend = close[i] < ema20_1w_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-day average volume
        avg_volume = np.mean(volume[i-20:i])
        volume_spike = volume[i] > avg_volume * 2.0
        
        # 20-day highest high and lowest low
        hh_20 = np.max(high[i-20:i])
        ll_20 = np.min(low[i-20:i])
        
        if position == 0:
            # Wyckoff Spring: price breaks below 20-day low (trap bears) then closes back above it with volume
            spring_long = (low[i] < ll_20) and (close[i] > ll_20) and volume_spike and uptrend
            
            # Wyckoff Upthorn: price breaks above 20-day high (trap bulls) then closes back below it with volume
            upthorn_short = (high[i] > hh_20) and (close[i] < hh_20) and volume_spike and downtrend
            
            if spring_long:
                signals[i] = 0.25
                position = 1
            elif upthorn_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 10-day low OR weekly trend turns down
            exit_low = np.min(low[i-10:i])
            if close[i] < exit_low or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 10-day high OR weekly trend turns up
            exit_high = np.max(high[i-10:i])
            if close[i] > exit_high or uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals