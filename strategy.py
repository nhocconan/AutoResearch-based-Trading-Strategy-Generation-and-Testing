#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Extreme
Hypothesis: KAMA (Kaufman Adaptive Moving Average) identifies trend direction with lag reduction,
while RSI extremes (>70 or <30) provide mean-reversion entries in the direction of the KAMA trend.
Combined with volume confirmation (>1.5x 20-day average) and weekly trend filter (EMA50),
this strategy captures sustainable moves while avoiding chop. Works in bull/bear by following
weekly trend direction. Designed for low trade frequency (<20/year) to minimize fee drag.
"""

name = "1d_KAMA_Trend_RSI_Extreme"
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
    volume = prices['volume'].values

    # Get weekly data ONCE before loop for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values

    # Calculate weekly EMA50 trend filter
    ema_50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)

    # Calculate daily KAMA trend (10-period ER, 2/30 SC)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # placeholder, will compute properly below
    # Recalculate volatility properly: sum of absolute changes over 10 days
    volatility_sum = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility_sum[i] = np.sum(np.abs(np.diff(close[i-9:i+1], prepend=close[i-9])))
    # Avoid division by zero
    er = np.where(volatility_sum > 0, change / volatility_sum, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    # Align KAMA to daily (already daily, but ensure alignment for consistency)
    kama_aligned = kama  # already aligned to daily

    # Calculate daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    # Volume confirmation: >1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        if (np.isnan(ema_50_weekly_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA (uptrend) + RSI > 70 (overbought in uptrend = continuation) + volume spike + weekly uptrend
            if (close[i] > kama[i] and 
                rsi[i] > 70 and 
                volume_spike[i] and 
                close[i] > ema_50_weekly_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA (downtrend) + RSI < 30 (oversold in downtrend = continuation) + volume spike + weekly downtrend
            elif (close[i] < kama[i] and 
                  rsi[i] < 30 and 
                  volume_spike[i] and 
                  close[i] < ema_50_weekly_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below KAMA (trend change) or RSI < 50 (momentum loss)
            if close[i] < kama[i] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above KAMA (trend change) or RSI > 50 (momentum loss)
            if close[i] > kama[i] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals