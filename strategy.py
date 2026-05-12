#!/usr/bin/env python3
# 12h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Combine 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
# Camarilla levels provide mean-reversion boundaries; breaks above R3 or below S3 indicate strong momentum.
# Filtering by 1d EMA50 ensures trades align with long-term trend, reducing false breakouts.
# Volume confirmation adds conviction to breakouts.
# Designed for 12-37 trades/year per symbol, works in both bull and bear via trend filter.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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

    # Get 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R3 = C + 1.1*(H-L)/2, S3 = C - 1.1*(H-L)/2
    prev_1d_high = df_1d['high'].shift(1).values  # Previous bar
    prev_1d_low = df_1d['low'].shift(1).values
    prev_1d_close = df_1d['close'].shift(1).values

    # Calculate R3 and S3 levels
    r3 = prev_1d_close + 1.1 * (prev_1d_high - prev_1d_low) / 2
    s3 = prev_1d_close - 1.1 * (prev_1d_high - prev_1d_low) / 2

    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)

    # Volume confirmation: current volume > 1.5x average of last 4 periods (2 days)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]

        if position == 0:
            # LONG: Close breaks above R3 AND uptrend AND volume
            if close[i] > r3_aligned[i] and price_above_ema and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S3 AND downtrend AND volume
            elif close[i] < s3_aligned[i] and price_below_ema and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close falls back below R3 OR trend turns down
            if close[i] < r3_aligned[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close rises back above S3 OR trend turns up
            if close[i] > s3_aligned[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals