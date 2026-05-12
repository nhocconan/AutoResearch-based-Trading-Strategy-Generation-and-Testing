#!/usr/bin/env python3
"""
4h_RSI_Divergence_Trend_Filter_Volume
Hypothesis: Enter long on bullish RSI divergence (price makes lower low, RSI makes higher low) when price > 200 EMA and volume > 1.5x average. Enter short on bearish RSI divergence (price makes higher high, RSI makes lower high) when price < 200 EMA and volume > 1.5x average. Exit on opposite divergence or when price crosses 200 EMA. This strategy aims to capture reversals in both bull and bear markets by combining momentum divergence with trend and volume filters. Target: 20-40 trades/year.
Timeframe: 4h
"""

name = "4h_RSI_Divergence_Trend_Filter_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 210:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Calculate 200 EMA for trend filter
    close_s = pd.Series(close)
    ema_200 = close_s.ewm(span=200, adjust=False, min_periods=200).mean().values

    # Calculate RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Volume filter: current > 1.5x average of last 12 bars (2 days on 4h)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_filter = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):  # Start after RSI warmup
        if np.isnan(ema_200[i]) or np.isnan(rsi[i]) or np.isnan(volume_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Bullish divergence: price lower low, RSI higher low
            if i >= 2:
                price_lower_low = low[i] < low[i-1] and low[i-1] < low[i-2]
                rsi_higher_low = rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2]
                bullish_div = price_lower_low and rsi_higher_low
            else:
                bullish_div = False

            # Bearish divergence: price higher high, RSI lower high
            if i >= 2:
                price_higher_high = high[i] > high[i-1] and high[i-1] > high[i-2]
                rsi_lower_high = rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2]
                bearish_div = price_higher_high and rsi_lower_high
            else:
                bearish_div = False

            # LONG: bullish divergence + price > EMA200 + volume filter
            if bullish_div and close[i] > ema_200[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: bearish divergence + price < EMA200 + volume filter
            elif bearish_div and close[i] < ema_200[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: bearish divergence or price < EMA200
            if i >= 2:
                price_higher_high = high[i] > high[i-1] and high[i-1] > high[i-2]
                rsi_lower_high = rsi[i] < rsi[i-1] and rsi[i-1] < rsi[i-2]
                bearish_div = price_higher_high and rsi_lower_high
            else:
                bearish_div = False
            if bearish_div or close[i] < ema_200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: bullish divergence or price > EMA200
            if i >= 2:
                price_lower_low = low[i] < low[i-1] and low[i-1] < low[i-2]
                rsi_higher_low = rsi[i] > rsi[i-1] and rsi[i-1] > rsi[i-2]
                bullish_div = price_lower_low and rsi_higher_low
            else:
                bullish_div = False
            if bullish_div or close[i] > ema_200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals