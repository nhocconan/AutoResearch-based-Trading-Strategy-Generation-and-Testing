#!/usr/bin/env python3
# 6h_LongTermTrend_With_ShortTermPullback_Entry
# Hypothesis: Enter long when price pulls back to 20-period EMA during a strong weekly uptrend, confirmed by volume.
# Enter short when price bounces off 20-period EMA during a strong weekly downtrend, confirmed by volume.
# Uses weekly trend as primary filter to avoid counter-trend trades, with 6EMA pullback as entry signal.
# Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
# Low frequency due to strong trend requirement and pullback confirmation.

name = "6h_LongTermTrend_With_ShortTermPullback_Entry"
timeframe = "6h"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly trend: EMA50
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly trend to 6h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Short-term EMA for pullback entry (20-period on 6h)
    ema20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(ema20_6h[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price near 20EMA + weekly uptrend + volume confirmation
            # Allow small buffer: price within 0.5% of EMA20
            ema_distance = abs(close[i] - ema20_6h[i]) / ema20_6h[i]
            if (ema_distance < 0.005 and  # Within 0.5% of EMA20
                close[i] > ema20_6h[i] and   # Slightly above EMA (pullback complete)
                close[i] > ema50_1w_aligned[i] and  # Weekly uptrend
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price near 20EMA + weekly downtrend + volume confirmation
            elif (ema_distance < 0.005 and  # Within 0.5% of EMA20
                  close[i] < ema20_6h[i] and   # Slightly below EMA (pullback complete)
                  close[i] < ema50_1w_aligned[i] and  # Weekly downtrend
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 20EMA OR weekly trend turns down
            if close[i] < ema20_6h[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 20EMA OR weekly trend turns up
            if close[i] > ema20_6h[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals