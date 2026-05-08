#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA21 trend filter + volume confirmation
# The 1d Donchian(20) provides a clear breakout signal in both bull and bear markets.
# The 1w EMA21 acts as a strong trend filter to avoid false breakouts in ranging markets.
# Volume confirmation ensures breakouts have participation.
# This combination aims for low-frequency, high-conviction trades with minimal whipsaw.
# Targets 10-25 trades per year (~40-100 total over 4 years) to minimize fee drag.

name = "1d_Donchian20_1wEMA21_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate EMA21 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema21_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        ema21_val = ema21_1w_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + volume confirmation + 1w uptrend
            if close_val > high_20_val and vol_conf_val and close_val > ema21_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian + volume confirmation + 1w downtrend
            elif close_val < low_20_val and vol_conf_val and close_val < ema21_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian or trend turns down
            if close_val < low_20_val or close_val < ema21_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian or trend turns up
            if close_val > high_20_val or close_val > ema21_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals