#!/usr/bin/env python3
"""
6h_RSI200_Trend_Breakout_With_Volume
Hypothesis: Uses RSI(200) to define long-term trend (RSI>50 bullish, <50 bearish).
Enters long when price breaks above 6h RSI(14) from oversold (<30) in bullish trend,
or short when price breaks below 6h RSI(14) from overbought (>70) in bearish trend.
Volume confirmation ensures momentum. Designed for low frequency and works in both
bull/bear markets by aligning with long-term trend via RSI(200).
"""

name = "6h_RSI200_Trend_Breakout_With_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Long-term trend: RSI(200) on daily closes (updated only when daily bar closes)
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 200:
        return np.zeros(n)
    close_d = df_d['close'].values
    delta = np.diff(close_d, prepend=close_d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/200, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/200, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_200 = 100 - (100 / (1 + rs))
    rsi_200 = np.where(avg_loss == 0, 100, rsi_200)  # handle zero loss
    rsi_200 = np.where(avg_gain == 0, 0, rsi_200)    # handle zero gain
    rsi_200_aligned = align_htf_to_ltf(prices, df_d, rsi_200)

    # Entry signal: RSI(14) on 6h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14 = np.where(avg_loss == 0, 100, rsi_14)
    rsi_14 = np.where(avg_gain == 0, 0, rsi_14)

    # Volume confirmation: 24-period average (4 days of 6h data)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):  # Start from 24 for volume average
        rsi200_val = rsi_200_aligned[i]
        rsi14_val = rsi_14[i]
        vol_avg_val = vol_avg_24[i]

        if np.isnan(rsi200_val) or np.isnan(rsi14_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI14 crosses above 30 from below in bullish long-term trend (RSI200>50) with volume
            if (rsi_14[i-1] <= 30 and rsi14_val > 30 and 
                rsi200_val > 50 and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI14 crosses below 70 from above in bearish long-term trend (RSI200<50) with volume
            elif (rsi_14[i-1] >= 70 and rsi14_val < 70 and 
                  rsi200_val < 50 and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI14 crosses below 50 (momentum loss) or RSI200 turns bearish
            if (rsi14_val < 50 or rsi200_val < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI14 crosses above 50 (momentum loss) or RSI200 turns bullish
            if (rsi14_val > 50 or rsi200_val > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals