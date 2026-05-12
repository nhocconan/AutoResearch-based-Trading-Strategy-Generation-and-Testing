#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolatilityFilter
Hypothesis: Camarilla pivot point breakouts at R1/S1 levels with 4h trend filter and 1d volatility filter capture institutional order flow in both bull and bear markets.
Breakouts above R1 + 4h uptrend + low volatility = long; breakdowns below S1 + 4h downtrend + low volatility = short.
Uses Camarilla levels derived from previous day's OHLC for institutional reference points.
Target: 20-40 trades/year per symbol with disciplined risk management.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolatilityFilter"
timeframe = "1h"
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

    # Get 1d data for Camarilla calculation and volatility filter (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels (R1, S1) from previous day's OHLC
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    camarilla_range = high_1d - low_1d
    r1 = close_1d + 1.1 * camarilla_range / 12
    s1 = close_1d - 1.1 * camarilla_range / 12
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)

    # Get 4h data for trend filter (call once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    # 4h EMA50 for trend
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)

    # 1d volatility filter: ATR(14) < 20-period average ATR (low volatility regime)
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]  # first period
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_20_1d = pd.Series(tr_1d).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr_14_1d < atr_ma_20_1d  # True when volatility is below average
    volatility_filter_aligned = align_htf_to_ltf(prices, df_1d, volatility_filter)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):  # Start from 1 to align with previous day's levels
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or np.isnan(volatility_filter_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above R1 + 4h uptrend + low volatility
            if close[i] > r1_aligned[i] and close[i] > ema50_4h_aligned[i] and volatility_filter_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Close breaks below S1 + 4h downtrend + low volatility
            elif close[i] < s1_aligned[i] and close[i] < ema50_4h_aligned[i] and volatility_filter_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below S1 or 4h trend turns down
            if close[i] < s1_aligned[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Close crosses above R1 or 4h trend turns up
            if close[i] > r1_aligned[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals