#!/usr/bin/env python3
# 4h_Stochastic_Reversal_RiskOff
# Hypothesis: In BTC/ETH, extreme Stochastic readings (overbought/oversold) combined with risk-off sentiment (VIX proxy via BTC volatility) and 1d trend filter capture mean-reversion moves. Works in bull via oversold bounces and bear via overbought reversals. Target: 20-30 trades/year per symbol.

name = "4h_Stochastic_Reversal_RiskOff"
timeframe = "4h"
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

    # Get 1d data for trend filter and volatility (VIX proxy)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # 1d EMA50 for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # 14-period Stochastic Oscillator
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    # Avoid division by zero
    denominator = highest_high - lowest_low
    denominator = np.where(denominator == 0, 1, denominator)
    stoch = 100 * (close - lowest_low) / denominator

    # BTC volatility as risk-off proxy (20-period ATR of 1d returns)
    returns_1d = np.diff(np.log(close_1d), prepend=np.log(close_1d[0]))
    atr_returns = pd.Series(np.abs(returns_1d)).rolling(window=20, min_periods=20).mean().values
    atr_returns_aligned = align_htf_to_ltf(prices, df_1d, atr_returns)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(stoch[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(atr_returns_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Oversold (<20) + 1d uptrend + rising volatility (risk-off fading)
            if stoch[i] < 20 and close[i] > ema50_1d_aligned[i] and atr_returns_aligned[i] > atr_returns_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # SHORT: Overbought (>80) + 1d downtrend + rising volatility (risk-off)
            elif stoch[i] > 80 and close[i] < ema50_1d_aligned[i] and atr_returns_aligned[i] > atr_returns_aligned[i-1]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Stochastic crosses above 50 or 1d trend turns down
            if stoch[i] > 50 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Stochastic crosses below 50 or 1d trend turns up
            if stoch[i] < 50 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals