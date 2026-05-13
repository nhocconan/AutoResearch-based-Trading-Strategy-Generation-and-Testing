#!/usr/bin/env python3
# 1d_Camarilla_R1_S1_Breakout_1wTrend
# Hypothesis: On the daily timeframe, enter long when price breaks above Camarilla R1 level with weekly uptrend and volume spike, enter short when price breaks below S1 level with weekly downtrend and volume spike.
# Camarilla levels provide precise intraday support/resistance derived from previous day's price action.
# Weekly trend filter ensures alignment with higher timeframe momentum, reducing false breakouts in choppy markets.
# Volume spike confirms institutional participation in the breakout.
# Works in bull markets (breakouts above R1 in uptrend) and bear markets (breakdowns below S1 in downtrend).
# Low frequency due to requirement of daily Camarilla breakout + weekly trend alignment + volume confirmation.

name = "1d_Camarilla_R1_S1_Breakout_1wTrend"
timeframe = "1d"
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
    
    # Daily high, low, close for Camarilla calculation
    high_1d = high
    low_1d = low
    close_1d = close
    
    # Calculate previous day's Camarilla levels
    # Using rolling window of 1 day shifted by 1 to get previous day's values
    prev_high = pd.Series(high_1d).shift(1)
    prev_low = pd.Series(low_1d).shift(1)
    prev_close = pd.Series(close_1d).shift(1)
    
    # Camarilla calculation
    R1 = prev_close + (prev_high - prev_low) * 1.0 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.0 / 12
    
    # Weekly trend: EMA50 on weekly close
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to daily timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike: volume > 2.0 * 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(R1[i]) or 
            np.isnan(S1[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R1 + weekly uptrend + volume spike
            if close[i] > R1[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S1 + weekly downtrend + volume spike
            elif close[i] < S1[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S1 or trend reversal
            if close[i] < S1[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R1 or trend reversal
            if close[i] > R1[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals