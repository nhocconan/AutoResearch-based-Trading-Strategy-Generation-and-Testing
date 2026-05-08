#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 12-hour trend filter and volume confirmation
# Long when price breaks above 4h Donchian Upper(20) + 12h EMA(50) uptrend + volume spike
# Short when price breaks below 4h Donchian Lower(20) + 12h EMA(50) downtrend + volume spike
# Donchian channels capture breakouts with clear structure
# 12h trend filter ensures alignment with higher timeframe momentum
# Volume spike confirms institutional participation
# Targets 75-200 total trades over 4 years (19-50/year) to avoid fee drag

name = "4h_Donchian20_12hTrend_Volume"
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
    
    # Get 12h data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_12h_val = ema50_12h_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price > upper + 12h uptrend + volume spike
            if close[i] > upper and close[i] > ema50_12h_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price < lower + 12h downtrend + volume spike
            elif close[i] < lower and close[i] < ema50_12h_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < lower OR 12h trend turns down
            if close[i] < lower or close[i] < ema50_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > upper OR 12h trend turns up
            if close[i] > upper or close[i] > ema50_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals