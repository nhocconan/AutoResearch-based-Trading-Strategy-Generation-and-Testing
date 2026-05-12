#!/usr/bin/env python3
# 6h_1D_1W_Camarilla_R3S3_Breakout_Trend_Filter
# Hypothesis: Breakouts at weekly Camarilla R3/S3 levels with daily trend filter and volume confirmation.
# In bull markets: buy R3 breakouts when daily trend is up. In bear markets: sell S3 breakdowns when daily trend is down.
# Uses weekly structure for higher-probability breaks and daily trend to avoid counter-trend trades.
# Volume confirmation reduces false breakouts. Designed for 6h to limit trade frequency (target: 12-37/year).

name = "6h_1D_1W_Camarilla_R3S3_Breakout_Trend_Filter"
timeframe = "6h"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Daily EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Get weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate Camarilla levels from previous weekly OHLC
    prev_weekly_close = df_1w['close'].shift(1).values
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values

    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_weekly_close + (prev_weekly_high - prev_weekly_low) * 1.1 / 4
    camarilla_s3 = prev_weekly_close - (prev_weekly_high - prev_weekly_low) * 1.1 / 4

    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from daily EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]

        if position == 0:
            # LONG: Break above weekly Camarilla R3 in uptrend with volume confirmation
            if (close[i] > camarilla_r3_aligned[i] and uptrend and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly Camarilla S3 in downtrend with volume confirmation
            elif (close[i] < camarilla_s3_aligned[i] and downtrend and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters weekly range (below R3) or trend reversal
            if close[i] < camarilla_r3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters weekly range (above S3) or trend reversal
            if close[i] > camarilla_s3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals