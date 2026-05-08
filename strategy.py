#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian(20) breakout with weekly trend filter and volume confirmation
# We go long when price breaks above the 20-day high with weekly EMA(34) uptrend and volume spike.
# We go short when price breaks below the 20-day low with weekly EMA(34) downtrend and volume spike.
# Uses 1d timeframe to target 7-25 trades/year, minimizing fee drag.
# Donchian channels provide clear breakout levels based on price extremes.
# Weekly trend filter ensures we trade with the higher timeframe momentum.
# Volume spike confirms institutional participation in the breakout.

name = "1d_Donchian20_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Donchian channels (20-day high/low) from daily data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1w_val = ema34_1w_aligned[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above upper band + weekly uptrend + volume spike
            if (not np.isnan(upper_band) and close[i] > upper_band and 
                close[i] > ema34_1w_val and vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band + weekly downtrend + volume spike
            elif (not np.isnan(lower_band) and close[i] < lower_band and 
                  close[i] < ema34_1w_val and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower band OR weekly trend turns down
            if (not np.isnan(lower_band) and close[i] < lower_band) or close[i] < ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper band OR weekly trend turns up
            if (not np.isnan(upper_band) and close[i] > upper_band) or close[i] > ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals