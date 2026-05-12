#!/usr/bin/env python3
"""
6h_RSI_Divergence_Strength_1dTrend
Hypothesis: Combine RSI divergence detection with price strength (close > open) and 1d EMA trend filter to capture high-probability reversals in both bull and bear markets. In bull markets, look for bullish RSI divergence during pullbacks; in bear markets, look for bearish RSI divergence during rallies. Volume confirmation filters weak signals.
"""

name = "6h_RSI_Divergence_Strength_1dTrend"
timeframe = "6h"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # RSI(14) calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        if np.isnan(rsi[i]) or np.isnan(rsi[i-1]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Bullish RSI divergence: price makes lower low, RSI makes higher low
            bull_div = (low[i] < low[i-1]) and (rsi[i] > rsi[i-1])
            # Bearish RSI divergence: price makes higher high, RSI makes lower high
            bear_div = (high[i] > high[i-1]) and (rsi[i] < rsi[i-1])
            
            # LONG: Bullish divergence + price strength (close > open) + 1d uptrend + volume spike
            if bull_div and close[i] > prices['open'].iloc[i] and close[i] > ema34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish divergence + price weakness (close < open) + 1d downtrend + volume spike
            elif bear_div and close[i] < prices['open'].iloc[i] and close[i] < ema34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish divergence or price weakness or 1d trend turns down
            bear_div = (high[i] > high[i-1]) and (rsi[i] < rsi[i-1])
            if bear_div or close[i] < prices['open'].iloc[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish divergence or price strength or 1d trend turns up
            bull_div = (low[i] < low[i-1]) and (rsi[i] > rsi[i-1])
            if bull_div or close[i] > prices['open'].iloc[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals