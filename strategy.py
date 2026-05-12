#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_Filter
Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) from 1d to determine trend direction and RSI(14) from 12h for mean-reversion entries within the trend.
In bull markets (price > KAMA), go long when RSI < 30 (oversold pullback). In bear markets (price < KAMA), go short when RSI > 70 (overbought bounce).
KAMA adapts to market noise, reducing whipsaws in choppy conditions. RSI provides timely entries during pullbacks.
Targets 12-30 trades/year by requiring trend alignment and extreme RSI readings.
"""

name = "12h_KAMA_Direction_RSI_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_len=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    close_series = pd.Series(close)
    change = abs(close_series - close_series.shift(er_len))
    volatility = abs(close_series.diff()).rolling(window=er_len, min_periods=1).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = [np.nan] * len(close)
    for i in range(len(close)):
        if i == 0:
            kama[i] = close[i]
        elif np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    return np.array(kama)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for KAMA trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Calculate KAMA on daily close
    kama_1d = calculate_kama(df_1d['close'].values, er_len=10, fast=2, slow=30)
    
    # Align KAMA to 12h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)

    # Get 12h data for RSI entry signal ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)

    # Calculate RSI on 12h close
    rsi_period = 14
    delta = pd.Series(df_12h['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Align RSI to 12h timeframe (already aligned since same timeframe)
    rsi_aligned = rsi

    # Volume confirmation: current volume > 1.2x average of last 4 periods
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_ok = volume > (1.2 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine trend direction from KAMA
        price_above_kama = close[i] > kama_1d_aligned[i]
        price_below_kama = close[i] < kama_1d_aligned[i]

        if position == 0:
            # LONG: price above KAMA (bullish trend) + RSI oversold + volume
            if price_above_kama and rsi_aligned[i] < 30 and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price below KAMA (bearish trend) + RSI overbought + volume
            elif price_below_kama and rsi_aligned[i] > 70 and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls below KAMA OR RSI overbought (take profit)
            if price_below_kama or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above KAMA OR RSI oversold (take profit)
            if price_above_kama or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals