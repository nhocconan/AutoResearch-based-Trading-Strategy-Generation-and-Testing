#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and 12h volume confirmation.
# Long when price breaks above 20-period high with 1d ATR ratio < 0.8 (low volatility regime) and 12h volume > 1.8x 20-period average.
# Short when price breaks below 20-period low with 1d ATR ratio < 0.8 and 12h volume > 1.8x 20-period average.
# Exit on opposite Donchian level (20-period low for longs, 20-period high for shorts).
# Uses ATR regime to avoid whipsaw in high volatility, volume confirmation for breakout validity.
# Target: 60-120 total trades over 4 years (15-30/year) to balance opportunity and fee drag.

name = "12h_Donchian20_Breakout_1dATRRegime_12hVolumeConfirm"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 12h Indicators (LTF) ---
    # 12h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume confirmation: > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_12h = volume > (1.8 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR(14) for volatility regime
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align with 1d index
    atr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1d ATR ratio: current ATR / 50-period average ATR (regime filter)
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_ma_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Low volatility regime: ATR ratio < 0.8 (below average volatility)
    low_vol_regime = atr_ratio_aligned < 0.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after Donchian warmup
        # Skip if missing data
        if (np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(volume_confirm_12h[i]) or
            np.isnan(low_vol_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 20-period high + low vol regime + volume confirmation
            if (close[i] > high_20[i] and 
                low_vol_regime[i] and 
                volume_confirm_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 20-period low + low vol regime + volume confirmation
            elif (close[i] < low_20[i] and 
                  low_vol_regime[i] and 
                  volume_confirm_12h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 20-period low
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 20-period high
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals