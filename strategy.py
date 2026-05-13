#!/usr/bin/env python3
"""
1D_KAMA_Trend_With_Chop_Filter
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets.
Combines KAMA trend filter with Choppiness Index regime filter to avoid whipsaws in ranging markets.
Uses weekly trend (EMA34) for higher timeframe confirmation and volume spike for institutional confirmation.
Designed for low-frequency, high-quality signals to minimize fee drag on 1d timeframe.
"""

name = "1D_KAMA_Trend_With_Chop_Filter"
timeframe = "1d"
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

    # KAMA efficiency ratio and smoothing constants
    def kama(close, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close)).cumsum()
        volatility = volatility - np.concatenate([[0], volatility[:-1]])
        er = np.zeros_like(close)
        for i in range(len(close)):
            if volatility[i] > 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama_out = np.zeros_like(close)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out

    # Choppiness Index
    def chop(high, low, close, window=14):
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
        tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        for i in range(window, len(close)):
            atr[i] = np.mean(tr[i-window+1:i+1])
        sumatr = np.zeros_like(close)
        for i in range(window-1, len(close)):
            sumatr[i] = np.sum(atr[i-window+1:i+1])
        hh = np.zeros_like(close)
        ll = np.zeros_like(close)
        for i in range(window-1, len(close)):
            hh[i] = np.max(high[i-window+1:i+1])
            ll[i] = np.min(low[i-window+1:i+1])
        chop_val = np.zeros_like(close)
        for i in range(window-1, len(close)):
            if hh[i] - ll[i] != 0:
                chop_val[i] = 100 * np.log10(sumatr[i] / (hh[i] - ll[i])) / np.log10(window)
            else:
                chop_val[i] = 50
        return chop_val

    # Get weekly data for higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Calculate indicators
    kama_val = kama(close, 10, 2, 30)
    chop_val = chop(high, low, close, 14)
    # Volume spike: volume > 2.0 * 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup for KAMA and CHOP
        # Skip if any required value is NaN
        if (np.isnan(kama_val[i]) or 
            np.isnan(chop_val[i]) or
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA (uptrend), Chop < 61.8 (trending market), above weekly EMA34, volume spike
            if close[i] > kama_val[i] and chop_val[i] < 61.8 and close[i] > ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA (downtrend), Chop < 61.8 (trending market), below weekly EMA34, volume spike
            elif close[i] < kama_val[i] and chop_val[i] < 61.8 and close[i] < ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or Chop > 61.8 (ranging market) or trend turns bearish
            if close[i] < kama_val[i] or chop_val[i] > 61.8 or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or Chop > 61.8 (ranging market) or trend turns bullish
            if close[i] > kama_val[i] or chop_val[i] > 61.8 or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals