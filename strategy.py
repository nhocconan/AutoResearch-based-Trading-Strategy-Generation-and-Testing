#!/usr/bin/env python3
# 4h_Donchian20_Breakout_1dTrend_Volume_Regime
# Hypothesis: Price breaks Donchian(20) channel with 1d trend filter and volume confirmation.
# Long when price breaks above upper band in 1d uptrend with volume spike.
# Short when price breaks below lower band in 1d downtrend with volume spike.
# Uses ADX(14) < 25 as range filter to avoid whipsaws in low volatility.
# Designed for low trade frequency (<50/year) to minimize fee drag and work in both bull/bear markets.

name = "4h_Donchian20_Breakout_1dTrend_Volume_Regime"
timeframe = "4h"
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

    # Get 1d data for trend filter and regime
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d ADX(14) for regime filter (avoid low volatility whipsaws)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), np.abs(high_1d[1:] - low_1d[:-1]))
    tr = np.concatenate([[np.nan], np.maximum(tr1, tr2)])
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian(20) channels on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > upper Donchian + 1d uptrend + volume spike + ADX < 25 (avoid chop)
            if close[i] > highest_high[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i] and adx_aligned[i] < 25:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < lower Donchian + 1d downtrend + volume spike + ADX < 25
            elif close[i] < lowest_low[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i] and adx_aligned[i] < 25:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below lower Donchian or trend reversal
            if close[i] < lowest_low[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above upper Donchian or trend reversal
            if close[i] > highest_high[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals