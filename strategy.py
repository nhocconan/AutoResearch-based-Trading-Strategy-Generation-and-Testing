#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Enter long when price breaks above Camarilla R1 level during low volatility, with weekly uptrend and volume confirmation.
# Enter short when price breaks below Camarilla S1 level during low volatility, with weekly downtrend and volume confirmation.
# Camarilla levels provide precise support/resistance. Low volatility increases breakout validity.
# Weekly trend filter ensures alignment with higher timeframe momentum, reducing false breakouts.
# Works in bull (breakouts above R1 in uptrend) and bear (breakdowns below S1 in downtrend).
# Low frequency due to volatility and trend filters.

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Get daily data for Camarilla levels and volatility
    df_1d = get_htf_data(prices, '1d')
    
    # Daily Camarilla levels (based on previous day's range)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    R1 = np.zeros_like(close_1d)
    S1 = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        range_val = high_1d[i-1] - low_1d[i-1]
        if range_val > 0:
            R1[i] = close_1d[i-1] + range_val * 1.1 / 12
            S1[i] = close_1d[i-1] - range_val * 1.1 / 12
        else:
            R1[i] = close_1d[i-1]
            S1[i] = close_1d[i-1]
    
    # Daily volatility: ATR(14) normalized by price
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], 
                     np.maximum(np.abs(high_1d[1:] - close_1d[:-1]),
                                np.abs(low_1d[1:] - close_1d[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    volatility = atr14 / close_1d
    vol_ma = pd.Series(volatility).rolling(window=10, min_periods=10).mean().values
    low_volatility = volatility < vol_ma  # Low volatility regime
    
    # Weekly trend: EMA50
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    low_volatility_aligned = align_htf_to_ltf(prices, df_1d, low_volatility)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike: volume > 1.5 * 2-period average (1 day worth at 12h)
    vol_ma_2 = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    volume_spike = volume > 1.5 * vol_ma_2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(low_volatility_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R1 + low volatility + weekly uptrend + volume spike
            if close[i] > R1_aligned[i] and low_volatility_aligned[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S1 + low volatility + weekly downtrend + volume spike
            elif close[i] < S1_aligned[i] and low_volatility_aligned[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S1 OR trend reversal
            if close[i] < S1_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R1 OR trend reversal
            if close[i] > R1_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals