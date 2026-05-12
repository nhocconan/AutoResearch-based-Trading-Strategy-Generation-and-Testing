#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_Reversion
Hypothesis: In 4h timeframe, KAMA identifies adaptive trend direction while RSI(2) identifies short-term mean reversion opportunities. Enter long when trend is up (KAMA > price) and RSI is oversold (<15); enter short when trend is down (KAMA < price) and RSI is overbought (>85). Volume confirmation (>1.5x 20-period average) filters false signals. Designed to work in both bull and bear markets by combining trend-following with mean reversion on pullbacks. Targets 20-40 trades/year.
"""

name = "4h_KAMA_Trend_RSI_Reversion"
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

    # Get daily data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Calculate daily KAMA for trend
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # This needs correction
    # Recalculate volatility properly: sum of absolute changes over period
    volatility = pd.Series(close_1d).rolling(window=10, min_periods=10).apply(lambda x: np.sum(np.abs(np.diff(x))), raw=True).values
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]

    # Calculate daily RSI(2)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=2, min_periods=2).mean().values
    avg_loss = pd.Series(loss).rolling(window=2, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values for current 4h bar
        kama_1d_a = align_htf_to_ltf(prices, df_1d, kama)[i]
        rsi_1d_a = align_htf_to_ltf(prices, df_1d, rsi)[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(kama_1d_a) or np.isnan(rsi_1d_a) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Trend up (KAMA > price) and RSI oversold + volume surge
            if (kama_1d_a > close[i] and 
                rsi_1d_a < 15 and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Trend down (KAMA < price) and RSI overbought + volume surge
            elif (kama_1d_a < close[i] and 
                  rsi_1d_a > 85 and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend turns down or RSI overbought
            if (kama_1d_a < close[i] or rsi_1d_a > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend turns up or RSI oversold
            if (kama_1d_a > close[i] or rsi_1d_a < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals