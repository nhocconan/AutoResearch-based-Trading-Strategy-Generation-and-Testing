#!/usr/bin/env python3
# 6h_ChaikinMoneyFlow_Stochastic_1dTrend_Filter
# Hypothesis: Chaikin Money Flow (CMF) measures institutional money flow strength.
# Combine with Stochastic oscillator for overbought/oversold signals filtered by 1d trend.
# Long when CMF > 0.10 and Stochastic %K < 20 (oversold) and 1d EMA50 uptrend.
# Short when CMF < -0.10 and Stochastic %K > 80 (overbought) and 1d EMA50 downtrend.
# Exit when CMF crosses back towards zero or Stochastic reverses.
# This combination filters false signals and works in both bull/bear via trend filter.
# Target: 20-30 trades/year on 6h to minimize fee drag.

name = "6h_ChaikinMoneyFlow_Stochastic_1dTrend_Filter"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate Chaikin Money Flow (CMF) - 20 period
    # CMF = sum((close - low - (high - close)) / (high - low) * volume) / sum(volume)
    # Simplified: ((close - low) - (high - close)) / (high - low) * volume = (2*close - high - low)/(high - low) * volume
    mfm = ((2 * close - high - low) / (high - low)).where(high != low, 0)  # Money Flow Multiplier
    mfv = mfm * volume  # Money Flow Volume
    cmf = pd.Series(mfv).rolling(window=20, min_periods=20).sum() / pd.Series(volume).rolling(window=20, min_periods=20).sum()
    cmf = cmf.values

    # Calculate Stochastic Oscillator (14,3,3)
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low)).where(highest_high != lowest_low, 50)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean()
    k_percent = k_percent.values
    d_percent = d_percent.values

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(cmf[i]) or np.isnan(k_percent[i]) or np.isnan(d_percent[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: CMF > 0.10 (buying pressure) + Stochastic oversold (%K < 20) + 1d UPTREND
            if (cmf[i] > 0.10 and 
                k_percent[i] < 20 and
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: CMF < -0.10 (selling pressure) + Stochastic overbought (%K > 80) + 1d DOWNTREND
            elif (cmf[i] < -0.10 and 
                  k_percent[i] > 80 and
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CMF crosses below 0 or Stochastic becomes overbought
            if cmf[i] < 0 or k_percent[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CMF crosses above 0 or Stochastic becomes oversold
            if cmf[i] > 0 or k_percent[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals