#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and 4h volume confirmation.
# Long when price breaks above upper Donchian(20) with 1d ATR ratio < 0.8 (low volatility regime) and 4h volume > 1.5x 20-period average.
# Short when price breaks below lower Donchian(20) with 1d ATR ratio < 0.8 and 4h volume > 1.5x 20-period average.
# Exit on opposite Donchian level (lower for longs, upper for shorts).
# Uses discrete position sizing (0.30) to balance capture and risk.
# ATR regime filter avoids high-volatility choppy markets where breakouts fail.
# Works in bull/bear: Donchian provides structure, volume confirms momentum, ATR filter ensures conducive volatility.

name = "4h_Donchian20_Breakout_1dATRRegime_4hVolumeConfirm"
timeframe = "4h"
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
    
    # --- 4h Indicators (LTF) ---
    # Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume confirmation: > 1.5x 20-period average (tight filter)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_4h = volume > (1.5 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR(14)
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d ATR ratio: ATR(14) / ATR(50) to detect volatility regime
    atr_50_1d = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio_1d = atr_14_1d / atr_50_1d
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Regime filter: low volatility (ATR ratio < 0.8) for breakout reliability
    low_vol_regime = atr_ratio_1d_aligned < 0.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(volume_confirm_4h[i]) or
            np.isnan(low_vol_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian + low vol regime + volume confirmation
            if (close[i] > highest_high_20[i] and 
                low_vol_regime[i] and 
                volume_confirm_4h[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below lower Donchian + low vol regime + volume confirmation
            elif (close[i] < lowest_low_20[i] and 
                  low_vol_regime[i] and 
                  volume_confirm_4h[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian
            if close[i] < lowest_low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian
            if close[i] > highest_high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals