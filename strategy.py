#!/usr/bin/env python3
# 4h_Stochastic_RSI_Trend_Filter
# Hypothesis: Use Stochastic RSI for momentum extremes with 1d EMA100 trend filter and volume confirmation.
# Stochastic RSI below 10 indicates oversold conditions for longs, above 90 for overbought shorts.
# Only trade in direction of 1d EMA100 trend to avoid counter-trend whipsaws.
# Volume > 2x 20-period average confirms institutional interest.
# Designed for low trade frequency (<30/year) with high win rate in both bull and bear markets.

name = "4h_Stochastic_RSI_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def stochastic_rsi(close, rsi_length=14, stoch_length=14, k=3, d=3):
    """Calculate Stochastic RSI."""
    # Calculate RSI
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    rs = pd.Series(up).ewm(alpha=1/rsi_length, adjust=False).mean() / \
         pd.Series(down).ewm(alpha=1/rsi_length, adjust=False).mean()
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Stochastic of RSI
    rsi_min = pd.Series(rsi).rolling(window=stoch_length, min_periods=stoch_length).min()
    rsi_max = pd.Series(rsi).rolling(window=stoch_length, min_periods=stoch_length).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
    
    # Calculate K and D
    k_percent = pd.Series(stoch_rsi).rolling(window=k, min_periods=k).mean()
    d_percent = pd.Series(k_percent).rolling(window=d, min_periods=d).mean()
    
    return d_percent.values  # Return smoothed D line

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
    
    # Calculate 1d EMA100 for trend filter
    ema_100_1d = pd.Series(df_1d['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)

    # Calculate Stochastic RSI
    stoch_rsi = stochastic_rsi(close, 14, 14, 3, 3)

    # Volume filter: >2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup period
        # Skip if any required value is NaN
        if (np.isnan(ema_100_1d_aligned[i]) or np.isnan(stoch_rsi[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Stochastic RSI < 10 (oversold) + price above 1d EMA100 (uptrend) + volume spike
            if (stoch_rsi[i] < 10 and 
                close[i] > ema_100_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Stochastic RSI > 90 (overbought) + price below 1d EMA100 (downtrend) + volume spike
            elif (stoch_rsi[i] > 90 and 
                  close[i] < ema_100_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Stochastic RSI > 50 (loss of momentum) or price below 1d EMA100 (trend change)
            if (stoch_rsi[i] > 50 or close[i] < ema_100_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Stochastic RSI < 50 (loss of momentum) or price above 1d EMA100 (trend change)
            if (stoch_rsi[i] < 50 or close[i] > ema_100_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals