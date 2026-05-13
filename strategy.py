#!/usr/bin/env python3
# 4h_KAMA_Trend_With_RSI_Filter
# Hypothesis: KAMA adapts to market noise, providing a smooth trend line that whipsaws less in choppy markets.
# Combined with RSI(14) for overbought/oversold conditions and volume confirmation to avoid false signals.
# Designed for 20-40 trades/year to minimize fee drag while maintaining edge in bull and bear markets.

name = "4h_KAMA_Trend_With_RSI_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for KAMA trend filter and RSI
    df_1d = get_htf_data(prices, '1d')

    # KAMA parameters
    er_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)

    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(df_1d['close'], prepend=df_1d['close'][0]))
    volatility = np.abs(np.diff(df_1d['close'])).cumsum()
    volatility = volatility - np.concatenate([[0], volatility[:-1]])  # rolling sum
    volatility = pd.Series(volatility).rolling(window=er_period, min_periods=er_period).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    er = pd.Series(er).rolling(window=er_period, min_periods=er_period).mean().values

    # Calculate Smoothing Constant (SC)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # Calculate KAMA
    kama = np.zeros_like(df_1d['close'])
    kama[0] = df_1d['close'].iloc[0]
    for i in range(1, len(df_1d)):
        kama[i] = kama[i-1] + sc[i] * (df_1d['close'].iloc[i] - kama[i-1])
    kama = kama  # already array

    # Align KAMA to 4h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)

    # Daily RSI(14)
    delta = np.diff(df_1d['close'], prepend=df_1d['close'][0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA, RSI < 70 (not overbought), and volume spike
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] < 70 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA, RSI > 30 (not oversold), and volume spike
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] > 30 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or RSI > 70
            if close[i] < kama_aligned[i] or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or RSI < 30
            if close[i] > kama_aligned[i] or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals