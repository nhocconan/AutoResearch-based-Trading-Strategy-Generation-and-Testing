#!/usr/bin/env python3
# 12h_Donchian_Breakout_Volume_Filter
# Hypothesis: Price breaks Donchian(20) channel on 1d timeframe with volume confirmation and ATR volatility filter.
# Long when price breaks above upper band with volume spike and ATR > median ATR (avoid low volatility chop).
# Short when price breaks below lower band with volume spike and ATR > median ATR.
# Uses 1d HTF for Donchian channels to reduce noise and false breakouts, suitable for 12h execution.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend) by capturing strong moves.
# Target: 15-35 trades/year per symbol to minimize fee drag.

name = "12h_Donchian_Breakout_Volume_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian(20) channels: upper = max(high,20), lower = min(low,20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    upper = np.full_like(high_1d, np.nan)
    lower = np.full_like(low_1d, np.nan)
    
    for i in range(20, len(high_1d)):
        upper[i] = np.max(high_1d[i-20:i])
        lower[i] = np.min(low_1d[i-20:i])
    
    # Calculate ATR(14) for volatility filter
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - np.roll(close, 1)[1:])
    tr3 = np.abs(low_1d[1:] - np.roll(close, 1)[1:])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.insert(tr, 0, np.nan)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Median ATR for volatility regime filter
    atr_median = np.full_like(atr, np.nan)
    for i in range(50, len(atr)):  # 50-period lookback for median
        atr_median[i] = np.nanmedian(atr[i-50:i])
    
    # Align 1d indicators to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    atr_median_aligned = align_htf_to_ltf(prices, df_1d, atr_median)
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(atr_aligned[i]) or 
            np.isnan(atr_median_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > upper band + volume spike + ATR > median ATR (avoid low volatility)
            if close[i] > upper_aligned[i] and volume_spike[i] and atr_aligned[i] > atr_median_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < lower band + volume spike + ATR > median ATR
            elif close[i] < lower_aligned[i] and volume_spike[i] and atr_aligned[i] > atr_median_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below lower band or ATR drops below median (low volatility exit)
            if close[i] < lower_aligned[i] or atr_aligned[i] < atr_median_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above upper band or ATR drops below median
            if close[i] > upper_aligned[i] or atr_aligned[i] < atr_median_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals