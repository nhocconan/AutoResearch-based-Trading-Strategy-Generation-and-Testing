#!/usr/bin/env python3
# 6h_WilliamsFractal_Range_Breakout
# Hypothesis: In ranging markets (ADX < 20), Williams fractals act as support/resistance.
# Breakouts above/below recent fractals with volume confirmation capture momentum.
# Works in bull markets via upside breakouts, in bear via downside breakdowns.
# Uses 1-day Williams fractals for structure, 60-minute ADX for regime filter.
# Target: 15-25 trades/year.

name = "6h_WilliamsFractal_Range_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_fractal(high, low):
    """Returns bearish (1) and bullish (-1) fractal signals."""
    n = len(high)
    bearish = np.zeros(n)
    bullish = np.zeros(n)
    for i in range(2, n - 2):
        if (high[i] > high[i-1] and high[i] > high[i-2] and
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = 1
        if (low[i] < low[i-1] and low[i] < low[i-2] and
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = -1
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data ONCE before loop for fractals
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate Williams fractals on 1d
    bearish_fractal, bullish_fractal = williams_fractal(high_1d, low_1d)
    # Fractals need 2-bar confirmation delay after the center bar
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)

    # Get 60m data for ADX regime filter (equivalent to 4 bars on 6h chart)
    df_60m = get_htf_data(prices, '60m')
    high_60m = df_60m['high'].values
    low_60m = df_60m['low'].values
    close_60m = df_60m['close'].values

    # ADX calculation (14-period) on 60m
    plus_dm = np.where((high_60m[1:] - high_60m[:-1]) > (low_60m[:-1] - low_60m[1:]), 
                       np.maximum(high_60m[1:] - high_60m[:-1], 0), 0)
    minus_dm = np.where((low_60m[:-1] - low_60m[1:]) > (high_60m[1:] - high_60m[:-1]), 
                        np.maximum(low_60m[:-1] - low_60m[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])

    tr1 = high_60m[1:] - low_60m[1:]
    tr2 = np.abs(high_60m[1:] - close_60m[:-1])
    tr3 = np.abs(low_60m[1:] - close_60m[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values

    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_60m = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_60m_aligned = align_htf_to_ltf(prices, df_60m, adx_60m)

    # Volume spike: current > 2.0x average of last 6 bars (6 hours)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after sufficient warmup
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(adx_60m_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Range regime: ADX < 20
        if adx_60m_aligned[i] < 20:
            if position == 0:
                # LONG: price breaks above recent bearish fractal resistance + volume spike
                if bearish_fractal_aligned[i] == 1 and close[i] > high[i-1] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # SHORT: price breaks below recent bullish fractal support + volume spike
                elif bullish_fractal_aligned[i] == -1 and close[i] < low[i-1] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # EXIT LONG: price breaks below bullish fractal support or ADX trends up
                if bullish_fractal_aligned[i] == -1 or adx_60m_aligned[i] > 25:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # EXIT SHORT: price breaks above bearish fractal resistance or ADX trends up
                if bearish_fractal_aligned[i] == 1 or adx_60m_aligned[i] > 25:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # Trending regime: stay flat or exit
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0

    return signals