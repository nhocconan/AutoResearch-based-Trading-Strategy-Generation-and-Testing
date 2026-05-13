#!/usr/bin/env python3
# 4h_KAMA_Trend_RSI_Chop_Filter
# Hypothesis: In choppy markets (CHOP > 61.8), price tends to revert to KAMA; in trending markets (CHOP < 38.2), follow KAMA direction.
# Entry: Long when price crosses above KAMA in trending up OR crosses below KAMA in chop with RSI < 30.
# Short when price crosses below KAMA in trending down OR crosses above KAMA in chop with RSI > 70.
# Uses 1d trend filter for higher timeframe alignment. Targets 20-50 trades/year to minimize fee drag.

name = "4h_KAMA_Trend_RSI_Chop_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Calculate KAMA (20, 2, 30)
    close_s = pd.Series(close)
    change = abs(close_s.diff(1))
    volatility = change.rolling(window=20, min_periods=20).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Calculate RSI (14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values

    # Calculate Choppiness Index (14)
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0]-low[0], np.abs(high[0]-close[0]), np.abs(low[0]-close[0])])],
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((max_high - min_low) > 0,
                    100 * np.log10(np.sum(tr) / (max_high - min_low)) / np.log10(14),
                    50)
    chop = np.nan_to_num(chop, nan=50.0)

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)

    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(sma50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG conditions
            long_trend = close[i] > kama[i] and close[i] > sma50_1d_aligned[i] and chop[i] < 38.2
            long_chop = close[i] < kama[i] and rsi[i] < 30 and chop[i] > 61.8
            if (long_trend or long_chop) and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT conditions
            short_trend = close[i] < kama[i] and close[i] < sma50_1d_aligned[i] and chop[i] < 38.2
            short_chop = close[i] > kama[i] and rsi[i] > 70 and chop[i] > 61.8
            if (short_trend or short_chop) and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below KAMA or trend fails
            if close[i] < kama[i] or close[i] < sma50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above KAMA or trend fails
            if close[i] > kama[i] or close[i] > sma50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals