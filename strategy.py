#!/usr/bin/env python3
# 160113: 4h_Donchian_20_12hTrend_Volume_RSI32
# Hypothesis: Breakout above/below Donchian(20) channel with 12h EMA50 trend filter and volume confirmation captures strong trending moves. RSI(14) > 32 avoids buying into exhaustion and selling into panic. Works in bull/bear by following higher timeframe trend. Designed for low trade frequency (<400 total 4h trades) to minimize fee drag.

name = "4h_Donchian_20_12hTrend_Volume_RSI32"
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

    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')

    # Calculate 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)

    # Donchian(20) channel
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_filter = rsi_values > 32  # Avoid exhaustion/panic

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_confirm[i]) or 
            np.isnan(rsi_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian high + 12h uptrend + volume + RSI > 32
            if (close[i] > highest_20[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_confirm[i] and 
                rsi_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + 12h downtrend + volume + RSI > 32
            elif (close[i] < lowest_20[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_confirm[i] and 
                  rsi_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Donchian low (reversal)
            if close[i] < lowest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian high (reversal)
            if close[i] > highest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals