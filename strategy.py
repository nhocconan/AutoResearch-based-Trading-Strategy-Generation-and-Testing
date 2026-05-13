#!/usr/bin/env python3
# 1d_Keltner_Channel_Breakout_WeeklyTrend_VolumeFilter
# Hypothesis: Enter long when price closes above Keltner upper band (EMA20 + 2*ATR) during weekly uptrend with volume confirmation; enter short when price closes below Keltner lower band during weekly downtrend with volume confirmation.
# Keltner Channels adapt to volatility, reducing false breakouts in low-volatility periods. Weekly trend filter ensures alignment with higher timeframe momentum.
# Works in bull markets (breakouts above upper band in uptrend) and bear markets (breakdowns below lower band in downtrend).
# Low-frequency signals due to weekly trend filter and volume confirmation, minimizing fee drag.

name = "1d_Keltner_Channel_Breakout_WeeklyTrend_VolumeFilter"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # ATR for Keltner Channels (using True Range)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])  # First TR is inf to avoid look-ahead
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # EMA20 for Keltner Channel middle line
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channels
    upper = ema20 + 2 * atr
    lower = ema20 - 2 * atr
    
    # Weekly trend: EMA50
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma_20
    
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
            # LONG: Close > upper Keltner band + weekly uptrend + volume filter
            if close[i] > upper[i] and close[i] > ema50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < lower Keltner band + weekly downtrend + volume filter
            elif close[i] < lower[i] and close[i] < ema50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below EMA20 (middle line) OR weekly trend reversal
            if close[i] < ema20[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above EMA20 (middle line) OR weekly trend reversal
            if close[i] > ema20[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals