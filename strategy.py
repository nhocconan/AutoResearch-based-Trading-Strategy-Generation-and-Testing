#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour 20-period Donchian breakout with weekly trend filter and volume confirmation
# We go long when price breaks above upper Donchian channel with weekly EMA(50) uptrend and volume spike.
# We go short when price breaks below lower Donchian channel with weekly EMA(50) downtrend and volume spike.
# Uses 12h timeframe to target 12-37 trades/year, avoiding excessive frequency.
# Donchian channels provide clear trend-following breakout signals.
# Weekly trend filter ensures we trade with the higher timeframe momentum.
# Volume spike confirms institutional participation in the breakout.

name = "12h_Donchian20_WeeklyTrend_Volume"
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
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    weekly_close = df_1w['close'].values
    ema50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 12-period Donchian channels (for 12h timeframe, using 20 periods = ~10 days)
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_1w_val = ema50_1w_aligned[i]
        upper_donchian = donchian_high[i]
        lower_donchian = donchian_low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + weekly uptrend + volume spike
            if (not np.isnan(upper_donchian) and close[i] > upper_donchian and 
                close[i] > ema50_1w_val and vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian + weekly downtrend + volume spike
            elif (not np.isnan(lower_donchian) and close[i] < lower_donchian and 
                  close[i] < ema50_1w_val and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian OR weekly trend turns down
            if (not np.isnan(lower_donchian) and close[i] < lower_donchian) or close[i] < ema50_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian OR weekly trend turns up
            if (not np.isnan(upper_donchian) and close[i] > upper_donchian) or close[i] > ema50_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals