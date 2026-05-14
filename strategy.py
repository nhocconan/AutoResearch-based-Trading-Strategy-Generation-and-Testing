#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
# Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Weekly pivot (from 1w data) provides structural bias:
# - Price above weekly pivot (PP) = bullish bias (only allow longs)
# - Price below weekly pivot = bearish bias (only allow shorts)
# Donchian breakout captures momentum, volume spike confirms conviction.
# Designed to work in both bull and bear markets by aligning with higher timeframe structure.
# Targets 50-150 total trades over 4 years.

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
    # Volume spike: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Donchian Channel (20) - breakout levels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Shift by 1 to use prior bar's levels (no look-ahead)
    donchian_high = np.roll(donchian_high, 1)
    donchian_low = np.roll(donchian_low, 1)
    donchian_high[0] = np.nan
    donchian_low[0] = np.nan
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly Pivot Point (PP) from prior 1d bar: PP = (high + low + close) / 3
    # Using 1d data to compute daily pivot, then align to 6h
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(pp_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high AND price > weekly PP (bullish bias) AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > pp_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low AND price < weekly PP (bearish bias) AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < pp_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian low (breakdown) OR touches weekly PP (mean reversion to pivot)
            if close[i] < donchian_low[i] or close[i] < pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Donchian high (breakout) OR touches weekly PP (mean reversion to pivot)
            if close[i] > donchian_high[i] or close[i] > pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals