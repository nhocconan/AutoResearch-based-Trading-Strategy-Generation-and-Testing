#!/usr/bin/env python3
# 4h_Williams_R_Reversal_1dTrend_Volume
# Hypothesis: Williams %R identifies overbought/oversold extremes; long when %R crosses above -50 from oversold with volume spike and daily uptrend, short when %R crosses below -50 from overbought with volume spike and daily downtrend. Uses Williams %R(14) for mean reversion signals, daily EMA50 for trend filter, and volume > 1.5x 20-period average for confirmation. Designed for 4h timeframe to avoid overtrading. Works in bull markets via pullbacks in uptrends and in bear markets via bounces in downtrends.

name = "4h_Williams_R_Reversal_1dTrend_Volume"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Daily EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from daily EMA50
        price_above_daily_ema = close[i] > ema_50_1d_aligned[i]
        price_below_daily_ema = close[i] < ema_50_1d_aligned[i]

        if position == 0:
            # LONG: Williams %R crosses above -50 from oversold with volume spike and daily uptrend
            if i > 0 and not np.isnan(williams_r[i-1]) and williams_r[i-1] <= -80 and williams_r[i] > -50 and volume_ok[i] and price_above_daily_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -50 from overbought with volume spike and daily downtrend
            elif i > 0 and not np.isnan(williams_r[i-1]) and williams_r[i-1] >= -20 and williams_r[i] < -50 and volume_ok[i] and price_below_daily_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses below -50 or trend turns down
            if i > 0 and not np.isnan(williams_r[i-1]) and williams_r[i-1] >= -20 and williams_r[i] < -50 or not price_above_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses above -50 or trend turns up
            if i > 0 and not np.isnan(williams_r[i-1]) and williams_r[i-1] <= -80 and williams_r[i] > -50 or not price_below_daily_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals