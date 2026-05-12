#!/usr/bin/env python3
# 1d_RSI_Extreme_Trend_Filter
# Hypothesis: Daily RSI extremes with 1-week EMA trend filter and volume confirmation. 
# Long when RSI < 30 and price > weekly EMA20 with volume > 1.5x 20-day average. 
# Short when RSI > 70 and price < weekly EMA20 with volume spike. 
# Exit when RSI returns to neutral (40-60 range). Designed for low frequency (10-25 trades/year) to avoid fee drag.
# Works in bull/bear markets by combining mean reversion (RSI extremes) with trend filter (weekly EMA).

name = "1d_RSI_Extreme_Trend_Filter"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    # Calculate daily RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values

    # Calculate volume threshold (1.5x 20-day SMA)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI oversold (<30) with uptrend (price > weekly EMA20) and volume spike
            if (rsi_values[i] < 30 and 
                close[i] > ema20_1w_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought (>70) with downtrend (price < weekly EMA20) and volume spike
            elif (rsi_values[i] > 70 and 
                  close[i] < ema20_1w_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to neutral (>=40) or turns bearish
            if rsi_values[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral (<=60) or turns bullish
            if rsi_values[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals