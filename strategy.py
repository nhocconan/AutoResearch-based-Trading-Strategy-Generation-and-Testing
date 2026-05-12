#!/usr/bin/env python3
"""
1d_RSI2080_MeanReversion_4hTrendFilter
Hypothesis: On 1d timeframe, buy when RSI(14) < 20 (oversold) with 4h EMA50 trending up, sell when RSI > 80 (overbought) with 4h EMA50 trending down. Uses RSI extremes for mean reversion in ranging markets and 4h trend filter to avoid counter-trend trades. Targets 10-20 trades per year to minimize fee drag and improve generalization in bull/bear markets.
"""

name = "1d_RSI2080_MeanReversion_4hTrendFilter"
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

    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values

    # Calculate RSI(14) on daily closes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    def rsi_wilder(gain, loss, period=14):
        avg_gain = np.zeros_like(gain)
        avg_loss = np.zeros_like(loss)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        for i in range(period+1, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = rsi_wilder(gain, loss, 14)
    
    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if np.isnan(rsi[i]) or np.isnan(ema50_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI < 20 (oversold) + 4h uptrend
            if rsi[i] < 20 and close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI > 80 (overbought) + 4h downtrend
            elif rsi[i] > 80 and close[i] < ema50_4h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 60 (overbought threshold) OR trend turns down
            if rsi[i] > 60 or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 40 (oversold threshold) OR trend turns up
            if rsi[i] < 40 or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals