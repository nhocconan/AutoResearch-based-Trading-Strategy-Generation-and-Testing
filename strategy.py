#!/usr/bin/env python3
# 4h_Camarilla_R1S1_Breakout_1dTrend_VolumeS
# Hypothesis: Camarilla pivot levels from 1-day timeframe provide strong support/resistance.
# Breakout above R1 or below S1 with volume confirmation and 1-day EMA50 trend filter.
# Works in bull markets (breakouts with trend) and bear markets (mean reversion at extremes).
# Target: 20-40 trades/year to minimize fee drag.

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
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

    # Calculate Camarilla levels from daily timeframe (using previous day's data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    # Use previous day's OHLC to avoid look-ahead
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    prev_range = prev_high - prev_low

    # Camarilla levels
    R1 = prev_close + (prev_range * 1.1 / 12)
    S1 = prev_close - (prev_range * 1.1 / 12)

    # Align to 4h timeframe (waits for daily close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)

    # 1-day EMA50 trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R1 with volume spike and price above EMA50 (uptrend)
            if close[i] > R1_aligned[i] and volume_spike[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with volume spike and price below EMA50 (downtrend)
            elif close[i] < S1_aligned[i] and volume_spike[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below S1 (mean reversion) or loses trend
            if close[i] < S1_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above R1 (mean reversion) or loses trend
            if close[i] > R1_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals