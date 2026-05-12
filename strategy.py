#!/usr/bin/env python3
# 4h_Stochastic_Bollinger_Trend_Reversal
# Hypothesis: In ranging markets, price reverses at Bollinger Bands with Stochastic oversold/overbought conditions. In trending markets, follow 12h EMA50 trend. Volume confirms momentum. Works in both bull/bear by adapting to volatility and trend. Target: 25-40 trades/year.

name = "4h_Stochastic_Bollinger_Trend_Reversal"
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
    volume = prices['volume'].values

    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')

    # Calculate 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)

    # Bollinger Bands (20, 2.0) on 4h
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2.0 * std_20
    bb_lower = sma_20 - 2.0 * std_20

    # Stochastic Oscillator (14,3,3)
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = pd.Series(stoch_k).rolling(window=3, min_periods=3).mean().values

    # Volume filter: current > 1.3x average of last 20 bars
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(stoch_k[i]) or 
            np.isnan(stoch_d[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price at/below BB Lower + Stochastic oversold + 12h EMA50 uptrend + volume
            if (close[i] <= bb_lower[i] and 
                stoch_k[i] < 20 and 
                stoch_d[i] < 20 and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at/above BB Upper + Stochastic overbought + 12h EMA50 downtrend + volume
            elif (close[i] >= bb_upper[i] and 
                  stoch_k[i] > 80 and 
                  stoch_d[i] > 80 and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above SMA20 (mean reversion) OR Stochastic overbought
            if close[i] > sma_20[i] or stoch_k[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below SMA20 OR Stochastic oversold
            if close[i] < sma_20[i] or stoch_k[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals