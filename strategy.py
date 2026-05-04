#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume spike confirmation
# Uses Donchian channel from 4h for structure, 1d ATR normalized for volatility regime,
# and volume spike for confirmation. Designed for 20-30 trades/year to minimize fee drag.
# Works in bull markets via upside breakouts and in bear markets via downside breakdowns.
# ATR regime filter avoids whipsaws in low volatility and captures expansion phases.

name = "4h_Donchian20_1dATRRegime_VolumeSpike"
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
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d True Range and ATR(14) from prior completed 1d bar
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_1d_shifted = np.roll(atr14_1d, 1)
    atr14_1d_shifted[0] = np.nan
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d_shifted)
    
    # Calculate 4h Donchian(20) channels from prior completed 4h bar
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    high_max_20_shifted = np.roll(high_max_20, 1)
    low_min_20_shifted = np.roll(low_min_20, 1)
    high_max_20_shifted[0] = np.nan
    low_min_20_shifted[0] = np.nan
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(high_max_20_shifted[i]) or np.isnan(low_min_20_shifted[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ATR regime filter: only trade when ATR is above its 50-period EMA (volatility expansion)
        atr_ema_50 = pd.Series(atr14_1d_aligned).ewm(span=50, adjust=False, min_periods=50).mean().values
        volatile_regime = atr14_1d_aligned[i] > atr_ema_50[i] if not np.isnan(atr_ema_50[i]) else False
        
        if position == 0:
            # Long conditions: break above Donchian upper band AND volatility regime AND volume spike
            if close[i] > high_max_20_shifted[i] and volatile_regime and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian lower band AND volatility regime AND volume spike
            elif close[i] < low_min_20_shifted[i] and volatile_regime and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower band
            if close[i] < low_min_20_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper band
            if close[i] > high_max_20_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals