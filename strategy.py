#!/usr/bin/env python3
# 6h_WilliamsAlligator_1dTrend_Volume
# Hypothesis: Williams Alligator (13,8,5 SMAs) defines trend structure; trade in direction of 1d EMA34 trend when price is aligned with Alligator jaws (8 SMA) and confirmed by volume spikes. Works in trending markets by capturing momentum, avoids whipsaws via trend alignment. Target: 15-25 trades/year.

name = "6h_WilliamsAlligator_1dTrend_Volume"
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

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Williams Alligator on 6h: Jaw (13), Teeth (8), Lips (5) - all SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values

    # Volume spike: current > 2.0x average of last 12 bars (3 days)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):  # Start after Alligator jaw warmup
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above jaws (uptrend alignment) + 1d EMA34 uptrend + volume spike
            if (close[i] > jaw[i] and 
                close[i] > teeth[i] and 
                close[i] > lips[i] and  # All aligned bullish
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and  # 1d uptrend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below jaws (downtrend alignment) + 1d EMA34 downtrend + volume spike
            elif (close[i] < jaw[i] and 
                  close[i] < teeth[i] and 
                  close[i] < lips[i] and  # All aligned bearish
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and  # 1d downtrend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below teeth (8 SMA) or 1d trend breaks
            if close[i] < teeth[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above teeth (8 SMA) or 1d trend breaks
            if close[i] > teeth[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals