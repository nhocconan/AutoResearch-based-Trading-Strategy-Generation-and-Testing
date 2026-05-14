#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation.
# Uses Donchian channel (20-period high/low) from 6h for breakout structure,
# weekly pivot (from 1d data) for medium-term bias, and volume spike for conviction.
# Only takes longs when price > weekly pivot and shorts when price < weekly pivot.
# Designed to capture strong breakouts aligned with weekly momentum while avoiding
# counter-trend signals. Targets 12-37 trades/year per symbol.

name = "6h_Donchian20_Breakout_WeeklyPivot_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 6h Indicators (LTF) ---
    # Donchian channel (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # --- 1d Indicators (HTF) for weekly pivot ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly pivot levels from prior 1 week (5 trading days)
    # Using prior 5-day high/low/close for weekly pivot calculation
    highest_5d = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    lowest_5d = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    close_5d = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot point (PP) = (High + Low + Close) / 3
    weekly_pp = (highest_5d + lowest_5d + close_5d) / 3.0
    
    # Align weekly pivot to 6h (wait for completed 1d bar)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1d, weekly_pp)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(weekly_pp_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine bias based on weekly pivot
        bullish_bias = close[i] > weekly_pp_aligned[i]
        bearish_bias = close[i] < weekly_pp_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above Donchian high AND bullish bias AND volume spike
            if close[i] > highest_20[i] and bullish_bias and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low AND bearish bias AND volume spike
            elif close[i] < lowest_20[i] and bearish_bias and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian low (breakdown)
            if close[i] < lowest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Donchian high (breakout)
            if close[i] > highest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals