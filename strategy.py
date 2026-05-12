#!/usr/bin/env python3
# 6h_Camarilla_R3S3_Breakout_12hTrend_Volume
# Hypothesis: Combine 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation.
# Camarilla levels provide mean-reversion boundaries; breaks above R3 or below S3 indicate strong momentum.
# Filtering by 12h EMA50 ensures trades align with medium-term trend, reducing false breakouts.
# Volume confirmation adds conviction to breakouts.
# Designed for 12-37 trades/year per symbol, works in both bull and bear via trend filter.

name = "6h_Camarilla_R3S3_Breakout_12hTrend_Volume"
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

    # Get 12h data for trend filter and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)

    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)

    # Calculate Camarilla levels from previous 12h bar
    # Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # R3 = C + 1.1*(H-L)/2, S3 = C - 1.1*(H-L)/2
    # Using previous 12h bar's high, low, close
    prev_12h_high = df_12h['high'].shift(1).values  # Previous bar
    prev_12h_low = df_12h['low'].shift(1).values
    prev_12h_close = df_12h['close'].shift(1).values

    # Calculate R3 and S3 levels
    r3 = prev_12h_close + 1.1 * (prev_12h_high - prev_12h_low) / 2
    s3 = prev_12h_close - 1.1 * (prev_12h_high - prev_12h_low) / 2

    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)

    # Volume confirmation: current volume > 1.5x average of last 4 periods (4 hours)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from 12h EMA50
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]

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