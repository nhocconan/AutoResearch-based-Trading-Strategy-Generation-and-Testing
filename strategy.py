#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and chop regime filter.
# Long when price breaks above 20-period Donchian high AND 1d volume > 1.5x 20-period average AND 1d chop > 61.8 (ranging market, mean revert long on breakout).
# Short when price breaks below 20-period Donchian low AND 1d volume > 1.5x 20-period average AND 1d chop > 61.8.
# Uses ATR-based trailing stop (2.0x) for exits. Designed for low-frequency, high-conviction trades in ranging markets where breakouts revert to mean.
# Target: 12-30 trades/year.

name = "12h_Donchian20_VolumeSpike_Chop_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for volume spike and chop regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period average volume on 1d
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    
    # Calculate Choppiness Index on 1d (14-period)
    # True Range
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(tr_sum_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop_1d = np.where(range_14 > 0, 100 * np.log10(tr_sum_14 / range_14) / np.log10(14), 50)
    chop_1d = np.nan_to_num(chop_1d, nan=50.0)
    
    # Align HTF arrays to 12h timeframe (wait for completed 1d bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate Donchian channels on 12h (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    
    for i in range(100, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above Donchian high + volume spike + chop > 61.8 (ranging)
            if close[i] > highest_20[i] and volume_spike_aligned[i] and chop_1d_aligned[i] > 61.8:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]
            # SHORT: Break below Donchian low + volume spike + chop > 61.8 (ranging)
            elif close[i] < lowest_20[i] and volume_spike_aligned[i] and chop_1d_aligned[i] > 61.8:
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]
            else:
                signals[i] = 0.0
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop (2.0x ATR) or Donchian mid-line reversion
            trailing_stop = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
            donchian_exit = close[i] < (highest_20[i] + lowest_20[i]) / 2  # Revert to midpoint
            if trailing_stop or donchian_exit:
                signals[i] = 0.0
                position = 0
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop (2.0x ATR) or Donchian mid-line reversion
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
            donchian_exit = close[i] > (highest_20[i] + lowest_20[i]) / 2  # Revert to midpoint
            if trailing_stop or donchian_exit:
                signals[i] = 0.0
                position = 0
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals