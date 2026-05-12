#!/usr/bin/env python3
# 4h_Weekly_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Breakouts above weekly Camarilla R3 or below S3 on 4h timeframe, filtered by 1d EMA50 trend and volume confirmation.
# Weekly levels provide strong support/resistance; breaks indicate momentum shifts. 1d EMA50 filters for medium-term trend alignment.
# Volume confirmation ensures breakout conviction. Designed for 20-50 trades/year per symbol, works in bull/bear via trend filter.

name = "4h_Weekly_Camarilla_R3S3_Breakout_1dTrend_Volume"
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

    # Get weekly data for Camarilla calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate weekly Camarilla R3 and S3 from previous week
    prev_1w_high = df_1w['high'].shift(1).values
    prev_1w_low = df_1w['low'].shift(1).values
    prev_1w_close = df_1w['close'].shift(1).values

    # Camarilla: R3 = C + 1.1*(H-L)/2, S3 = C - 1.1*(H-L)/2
    r3 = prev_1w_close + 1.1 * (prev_1w_high - prev_1w_low) / 2
    s3 = prev_1w_close - 1.1 * (prev_1w_high - prev_1w_low) / 2

    # Align weekly levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)

    # Volume confirmation: current volume > 1.5x average of last 4 periods (16 hours)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
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
            # LONG: Close breaks above weekly R3 AND uptrend AND volume
            if close[i] > r3_aligned[i] and price_above_ema and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below weekly S3 AND downtrend AND volume
            elif close[i] < s3_aligned[i] and price_below_ema and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close falls back below weekly R3 OR trend turns down
            if close[i] < r3_aligned[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close rises back above weekly S3 OR trend turns up
            if close[i] > s3_aligned[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals