#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA200 trend filter + volume confirmation
# The 1d Donchian breakout captures strong momentum moves in the direction of the weekly trend.
# Entry occurs when price breaks above/below the 20-day high/low, confirmed by 1w EMA200 slope and volume spike (>2x 20-day average).
# Exits when price reverses to the opposite Donchian band or trend weakens.
# This strategy targets 20-50 trades over 4 years (5-12/year) to minimize fee drag and works in both bull and bear markets by following the higher timeframe trend.

name = "1d_Donchian20_1wEMA200_Volume"
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
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate EMA200 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_slope = ema200_1w[1:] - ema200_1w[:-1]  # slope: positive = uptrend
    ema200_1w_slope = np.concatenate([[0], ema200_1w_slope])  # align length
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    ema200_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w_slope)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema200_1w_aligned[i]) or np.isnan(ema200_1w_slope_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        ema200_val = ema200_1w_aligned[i]
        ema200_slope = ema200_1w_slope_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: price breaks above 20-day high, volume confirmation, 1w uptrend
            if close[i] > high_20_val and vol_conf_val and ema200_slope > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 20-day low, volume confirmation, 1w downtrend
            elif close[i] < low_20_val and vol_conf_val and ema200_slope < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 20-day low or 1w trend turns down
            if close[i] < low_20_val or ema200_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 20-day high or 1w trend turns up
            if close[i] > high_20_val or ema200_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals