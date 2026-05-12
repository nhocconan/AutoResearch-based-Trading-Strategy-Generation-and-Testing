#!/usr/bin/env python3
"""
1d_Weekly_RSI_MeanReversion_1wTrend
Hypothesis: Weekly RSI extremes (RSI<25 for long, RSI>75 for short) combined with weekly trend filter (price above/below 20-week EMA) provides high-probability mean-reversion entries that work in both bull and bear markets. Uses daily timeframe with weekly RSI and EMA filters to avoid counter-trend trades. Low-frequency signals (target 10-30 trades/year) minimize fee drag.
"""
name = "1d_Weekly_RSI_MeanReversion_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')

    # Calculate weekly RSI(14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))

    # Calculate weekly EMA(20) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Align weekly indicators to daily timeframe
    rsi_14_aligned = align_htf_to_ltf(prices, df_1w, rsi_14)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)

    # Volume confirmation: >1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after EMA20 warmup
        if (np.isnan(rsi_14_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI oversold (<25) + price above weekly EMA20 (uptrend) + volume confirmation
            if (rsi_14_aligned[i] < 25 and 
                close[i] > ema_20_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought (>75) + price below weekly EMA20 (downtrend) + volume confirmation
            elif (rsi_14_aligned[i] > 75 and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to neutral (>45) or trend breaks (price below weekly EMA20)
            if (rsi_14_aligned[i] > 45 or 
                close[i] < ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral (<55) or trend breaks (price above weekly EMA20)
            if (rsi_14_aligned[i] < 55 or 
                close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals