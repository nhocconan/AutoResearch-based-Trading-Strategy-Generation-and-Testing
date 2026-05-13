#!/usr/bin/env python3
# 4h_RSI_MeanReversion_VolumeSpike_Filter
# Hypothesis: Mean reversion at extreme RSI levels with volume spike and price above/below 200 SMA captures reversals in both bull and bear markets. Works by fading extremes when momentum exhausts.

name = "4h_RSI_MeanReversion_VolumeSpike_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # RSI(14) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # 200 SMA for trend filter (avoid counter-trend trades)
    sma200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values

    # Volume confirmation: volume > 1.8x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):
        # Skip if any required value is NaN
        if (np.isnan(rsi[i]) or np.isnan(sma200[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI < 20 (oversold) + price above SMA200 + volume spike
            if (rsi[i] < 20 and 
                close[i] > sma200[i] and
                volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI > 80 (overbought) + price below SMA200 + volume spike
            elif (rsi[i] > 80 and 
                  close[i] < sma200[i] and
                  volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 60 (overbought threshold) or volume drops
            if rsi[i] > 60 or volume[i] < vol_avg_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 40 (oversold threshold) or volume drops
            if rsi[i] < 40 or volume[i] < vol_avg_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals