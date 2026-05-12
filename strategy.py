#!/usr/bin/env python3
"""
4h_Volume_Weighted_RSI_Trend
Hypothesis: Combine RSI(14) with volume-weighted price action and 4h EMA trend filter.
Long when: RSI < 30 (oversold) + price > VWAP(20) + close > EMA(50)
Short when: RSI > 70 (overbought) + price < VWAP(20) + close < EMA(50)
Volume confirmation requires current volume > 1.5x 20-period average.
Uses discrete position sizing (0.25) to minimize churn. Targets 20-40 trades/year.
Works in bull via mean reversion off support, in bear via selling into resistance.
"""

name = "4h_Volume_Weighted_RSI_Trend"
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

    # Calculate VWAP(20): typical price * volume cumulative / volume cumulative
    typical_price = (high + low + close) / 3
    vp = typical_price * volume
    cum_vp = pd.Series(vp).rolling(window=20, min_periods=20).sum()
    cum_vol = pd.Series(volume).rolling(window=20, min_periods=20).sum()
    vwap = (cum_vp / cum_vol).values
    vwap[:19] = np.nan  # First 19 values invalid

    # EMA(50) for trend filter
    close_s = pd.Series(close)
    ema_50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values

    # RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = (100 - (100 / (1 + rs))).values
    rsi[:13] = np.nan  # First 13 values invalid

    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        if (np.isnan(rsi[i]) or np.isnan(vwap[i]) or np.isnan(ema_50[i]) or 
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI oversold + price above VWAP + close above EMA50 + volume
            if rsi[i] < 30 and close[i] > vwap[i] and close[i] > ema_50[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought + price below VWAP + close below EMA50 + volume
            elif rsi[i] > 70 and close[i] < vwap[i] and close[i] < ema_50[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 50 OR close < EMA50
            if rsi[i] > 50 or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 50 OR close > EMA50
            if rsi[i] < 50 or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals