#!/usr/bin/env python3
# 1d_Keltner_Breakout_WeeklyTrend_Volume
# Hypothesis: Enter long when price breaks above Keltner upper band during 1uptrend (weekly EMA50) with volume confirmation; short when price breaks below lower band during weekly downtrend with volume confirmation. Exit when price crosses the EMA20 (middle band).
# Keltner channels use ATR, which adapts to volatility, reducing false breakouts in low volatility and capturing real breakouts in high volatility.
# Weekly trend filter ensures alignment with higher timeframe momentum, reducing false signals in choppy markets.
# Volume surge confirms institutional participation. Works in bull (breakouts in uptrend) and bear (breakdowns in downtrend).
# Low frequency due to trend and volume requirements.

name = "1d_Keltner_Breakout_WeeklyTrend_Volume"
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

    # Get weekly data for trend
    df_1w = get_htf_data(prices, '1w')
    
    # Keltner Channels (20, 2) on daily
    atr_period = 20
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper = ema20 + 2 * atr
    lower = ema20 - 2 * atr
    
    # Weekly trend: EMA50
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly trend to daily
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > upper band + weekly uptrend + volume spike
            if close[i] > upper[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < lower band + weekly downtrend + volume spike
            elif close[i] < lower[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below EMA20 (middle band)
            if close[i] < ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above EMA20 (middle band)
            if close[i] > ema20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals