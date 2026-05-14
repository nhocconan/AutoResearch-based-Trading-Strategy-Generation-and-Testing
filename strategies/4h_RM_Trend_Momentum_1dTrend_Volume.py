#!/usr/bin/env python3
# 4h_RM_Trend_Momentum_1dTrend_Volume
# Hypothesis: Combine Relative Momentum Index (RMI) with trend confirmation and volume spikes for RMI(14) extremes in the direction of the 1d EMA50 trend. Uses RMI > 70 for longs in uptrend and RMI < 30 for shorts in downtrend, filtered by 1d EMA50 and volume > 1.5x 20-period average. Designed to capture momentum bursts in both bull and bear markets with low trade frequency.

name = "4h_RM_Trend_Momentum_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def rmi(close, length=14):
    """Calculate Relative Momentum Index."""
    mom = close - np.roll(close, 1)
    mom[0] = 0
    up = np.where(mom > 0, mom, 0)
    down = np.where(mom < 0, -mom, 0)
    # RSI-like smoothing
    rup = pd.Series(up).ewm(span=length, adjust=False).mean()
    rdown = pd.Series(down).ewm(span=length, adjust=False).mean()
    rsi = 100 - (100 / (1 + rup / (rdown + 1e-10)))
    return rsi.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate RMI(14)
    rmi_val = rmi(close, 14)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rmi_val[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RMI > 70 (bullish momentum) + price above 1d EMA50 (uptrend) + volume spike
            if (rmi_val[i] > 70 and 
                close[i] > ema_50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: RMI < 30 (bearish momentum) + price below 1d EMA50 (downtrend) + volume spike
            elif (rmi_val[i] < 30 and 
                  close[i] < ema_50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RMI < 50 (loss of momentum) or price below 1d EMA50 (trend change)
            if (rmi_val[i] < 50 or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RMI > 50 (loss of momentum) or price above 1d EMA50 (trend change)
            if (rmi_val[i] > 50 or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals