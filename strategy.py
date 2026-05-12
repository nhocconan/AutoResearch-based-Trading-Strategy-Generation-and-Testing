#!/usr/bin/env python3
"""
6h_Russell2000_SMA_Trend_Momentum
Hypothesis: Combine 60-day SMA trend filter with 20-bar RSI momentum to capture sustained trends while avoiding chop. Russell2000 SMA (60) acts as a long-term trend filter (price > SMA60 = bullish bias, < SMA60 = bearish bias). RSI(20) momentum confirms entry direction with overbought/oversold thresholds. Works in both bull and bear markets by using trend-following entries in the direction of the long-term SMA. Uses volume confirmation to avoid false breaks. Targets 15-35 trades/year on 6h timeframe with disciplined risk management via trend reversal exits.
Timeframe: 6h
"""

name = "6h_Russell2000_SMA_Trend_Momentum"
timeframe = "6h"
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

    # Get daily data for SMA60 trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)

    # Calculate daily SMA60 for trend filter
    close_1d = df_1d['close'].values
    sma_60_1d = pd.Series(close_1d).rolling(window=60, min_periods=60).mean().values
    sma_60_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_60_1d)

    # RSI(20) momentum on 6h timeframe
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/20, adjust=False, min_periods=20).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/20, adjust=False, min_periods=20).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values

    # Volume confirmation: current > 1.8x average of last 4 bars
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (1.8 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        if (np.isnan(sma_60_1d_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price > daily SMA60 + RSI > 50 (bullish momentum) + volume spike
            if (close[i] > sma_60_1d_aligned[i] and 
                rsi_values[i] > 50 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price < daily SMA60 + RSI < 50 (bearish momentum) + volume spike
            elif (close[i] < sma_60_1d_aligned[i] and 
                  rsi_values[i] < 50 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < daily SMA60 (trend reversal)
            if close[i] < sma_60_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > daily SMA60 (trend reversal)
            if close[i] > sma_60_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals