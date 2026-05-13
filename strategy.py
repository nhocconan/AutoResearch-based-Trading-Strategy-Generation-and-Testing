#!/usr/bin/env python3
# 4h_TrendFollowing_Turtle_System
# Hypothesis: Turtle trading system adapted for 4h timeframe with 20/55-day breakouts filtered by 1-day ATR volatility regime.
# Works in bull/bear: long on 55-bar high breakout during low volatility, short on 20-bar low breakdown during low volatility.
# Uses 1-day ATR to filter regime - only trade when volatility is below median (calm markets) to avoid whipsaw.
# Designed for 15-25 trades/year to minimize fee drag while capturing major trends.

name = "4h_TrendFollowing_Turtle_System"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values

    # Get daily data for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')

    # Calculate 1-day ATR(20) for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First period
    tr3[0] = np.abs(low_1d[0] - close_1d[0])  # First period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_median = np.nanmedian(atr_20[~np.isnan(atr_20)]) if np.sum(~np.isnan(atr_20)) > 0 else 1.0
    low_volatility = atr_20 < atr_median  # Trade only in low volatility regime
    
    # Align volatility filter to 4h timeframe
    low_volatility_aligned = align_htf_to_ltf(prices, df_1d, low_volatility)

    # Calculate 55-period high and 20-period low for breakout signals
    high_55 = pd.Series(high).rolling(window=55, min_periods=55).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(55, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(high_55[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(low_volatility_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above 55-period high during low volatility
            if close[i] > high_55[i] and low_volatility_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below 20-period low during low volatility
            elif close[i] < low_20[i] and low_volatility_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Break below 20-period low (tighter stop for risk management)
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Break above 55-period high
            if close[i] > high_55[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals