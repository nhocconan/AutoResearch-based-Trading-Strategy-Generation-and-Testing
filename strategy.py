#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume confirmation
# Uses Donchian channel breakouts for structure, 1d ATR for volatility filter (avoid low-vol chop),
# and volume spike for confirmation. Designed for 20-30 trades/year to minimize fee drag.
# Works in bull markets via breakout continuations and in bear markets via breakdown continuations.
# The 1d ATR filter ensures we only trade when volatility is sufficient, reducing whipsaw in low-vol regimes.

name = "4h_Donchian20_1dATRFilter_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) from prior completed 1d bar
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_1d_shifted = np.roll(atr14_1d, 1)
    atr14_1d_shifted[0] = np.nan
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d_shifted)
    
    # Calculate Donchian(20) channels from prior completed 4h bar
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when 1d ATR > 0.5 * 20-period SMA of ATR (avoid extremely low vol)
        atr_sma_20 = pd.Series(atr14_1d_aligned).rolling(window=20, min_periods=20).mean().values
        vol_filter = atr14_1d_aligned[i] > (0.5 * atr_sma_20[i]) if not np.isnan(atr_sma_20[i]) else False
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND volume spike AND vol filter
            if close[i] > highest_high_20[i] and volume[i] > (2.0 * vol_ema_20[i]) and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower band AND volume spike AND vol filter
            elif close[i] < lowest_low_20[i] and volume[i] > (2.0 * vol_ema_20[i]) and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower band
            if close[i] < lowest_low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper band
            if close[i] > highest_high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals