#!/usr/bin/env python3
# 1d_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Camarilla pivot levels (R3/S3) on daily timeframe identify key support/resistance.
# Breakouts above R3 or breakdowns below S3 with volume confirmation and weekly trend alignment
# provide high-probability trades. Weekly trend filter avoids counter-trend trades in strong trends.
# Works in bull and bear markets by trading breakouts of significant daily levels with trend filter.
# Target: 10-25 trades/year per symbol (40-100 total over 4 years).

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "1d"
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

    # Get daily data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    # Calculate Camarilla levels (R3, S3) from previous day
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    # Shift by 1 to use previous day's levels
    camarilla_r3 = (df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low'])).shift(1).values
    camarilla_s3 = (df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low'])).shift(1).values

    # Align Camarilla levels to 1d timeframe (no extra delay needed as we use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)

    # Weekly trend filter: EMA34 on weekly timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)

    # Volume confirmation: current volume > 1.5x average of last 20 days
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Weekly trend filter
        price_above_weekly_ema = close[i] > ema_34_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_34_1w_aligned[i]

        if position == 0:
            # LONG: Price breaks above R3 with volume and weekly uptrend
            if (not np.isnan(camarilla_r3_aligned[i]) and 
                close[i] > camarilla_r3_aligned[i] and
                price_above_weekly_ema and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume and weekly downtrend
            elif (not np.isnan(camarilla_s3_aligned[i]) and 
                  close[i] < camarilla_s3_aligned[i] and
                  price_below_weekly_ema and volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 or weekly trend turns down
            if (not np.isnan(camarilla_s3_aligned[i]) and 
                close[i] < camarilla_s3_aligned[i]) or not price_above_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 or weekly trend turns up
            if (not np.isnan(camarilla_r3_aligned[i]) and 
                close[i] > camarilla_r3_aligned[i]) or not price_below_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals