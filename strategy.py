#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 12-hour EMA(50) trend filter and volume confirmation
# Go long when price breaks above 4h Donchian high with 12h EMA(50) uptrend and volume spike.
# Go short when price breaks below 4h Donchian low with 12h EMA(50) downtrend and volume spike.
# Donchian provides clear trend-following structure. 12h EMA filters for higher timeframe momentum.
# Volume spike confirms institutional participation. Designed for 20-50 trades/year on 4h timeframe.

name = "4h_Donchian20_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12-hour data once for EMA(50) trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 4h Donchian channels (20-period)
    # Donchian high = max(high over last 20 periods)
    # Donchian low = min(low over last 20 periods)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_12h_val = ema50_12h_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high + 12h EMA uptrend + volume spike
            if (not np.isnan(upper) and close[i] > upper and 
                close[i] > ema50_12h_val and vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + 12h EMA downtrend + volume spike
            elif (not np.isnan(lower) and close[i] < lower and 
                  close[i] < ema50_12h_val and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR 12h EMA turns down
            if (not np.isnan(lower) and close[i] < lower) or close[i] < ema50_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high OR 12h EMA turns up
            if (not np.isnan(upper) and close[i] > upper) or close[i] > ema50_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals