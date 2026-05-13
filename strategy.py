#!/usr/bin/env python3
# 6h_ADX_DMI_Trend_Filter_1dEMA89_Trend
# Hypothesis: Strong trending conditions identified by ADX > 25 and DMI crossover, filtered by 1d EMA89 trend, capture sustained moves while avoiding chop. Works in both bull and bear by only taking longs in uptrend and shorts in downtrend.

name = "6h_ADX_DMI_Trend_Filter_1dEMA89_Trend"
timeframe = "6h"
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

    # Calculate ADX and DMI
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values

    # 1d EMA89 for trend filter (load once, align)
    df_1d = get_htf_data(prices, '1d')
    ema89_1d = pd.Series(df_1d['close'].values).ewm(span=89, adjust=False, min_periods=89).mean().values
    ema89_1d_aligned = align_htf_to_ltf(prices, df_1d, ema89_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required value is NaN
        if (np.isnan(adx[i]) or np.isnan(di_plus[i]) or np.isnan(di_minus[i]) or 
            np.isnan(ema89_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: ADX > 25, DI+ > DI-, and price above 1d EMA89
            if (adx[i] > 25 and 
                di_plus[i] > di_minus[i] and
                close[i] > ema89_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: ADX > 25, DI- > DI+, and price below 1d EMA89
            elif (adx[i] > 25 and 
                  di_minus[i] > di_plus[i] and
                  close[i] < ema89_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: ADX < 20 or DI- > DI+ or price below 1d EMA89
            if (adx[i] < 20 or 
                di_minus[i] > di_plus[i] or
                close[i] < ema89_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: ADX < 20 or DI+ > DI- or price above 1d EMA89
            if (adx[i] < 20 or 
                di_plus[i] > di_minus[i] or
                close[i] > ema89_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals