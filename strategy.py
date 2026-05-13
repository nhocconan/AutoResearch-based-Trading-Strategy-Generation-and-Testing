#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
# Hypothesis: Enter long when price breaks above Camarilla R1 during volatility contraction in the direction of 1d EMA34 trend, confirmed by volume spike. Enter short when price breaks below S1 under same conditions.
# Camarilla levels provide statistically significant intraday support/resistance. Volatility contraction (low ATR) precedes breakouts. Volume surge confirms institutional participation.
# Trend filter ensures alignment with higher timeframe momentum, reducing false breakouts in choppy markets. Works in bull (breakouts above R1 in uptrend) and bear (breakdowns below S1 in downtrend).
# Low frequency due to volatility contraction requirement and strict volume confirmation.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
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

    # Get daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla levels (based on previous day's range)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    # R1 = close + (range * 1.1 / 12), S1 = close - (range * 1.1 / 12)
    r1 = close_1d + (range_1d * 1.1 / 12)
    s1 = close_1d - (range_1d * 1.1 / 12)
    
    # Daily trend: EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volatility contraction: ATR(7) < ATR(14) * 0.8 (low volatility)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.inf], tr1])
    atr7 = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    volatility_contraction = atr7 < (atr14 * 0.8)
    
    # Align daily indicators to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volatility_contraction_aligned = align_htf_to_ltf(prices, df_1d, volatility_contraction)
    
    # Volume spike: volume > 1.5 * 2-period average (1 day worth at 12h)
    vol_ma_2 = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    volume_spike = volume > 1.5 * vol_ma_2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volatility_contraction_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R1 + volatility contraction + daily uptrend + volume spike
            if close[i] > r1_aligned[i] and volatility_contraction_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S1 + volatility contraction + daily downtrend + volume spike
            elif close[i] < s1_aligned[i] and volatility_contraction_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S1 OR trend reversal
            if close[i] < s1_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R1 OR trend reversal
            if close[i] > r1_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals