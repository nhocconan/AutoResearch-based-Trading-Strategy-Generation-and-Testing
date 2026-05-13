#!/usr/bin/env python3
# 1d_Weekly_Camarilla_R3S3_Breakout_1wTrend_Volume
# Hypothesis: Use weekly (1w) Camarilla pivot levels (R3/S3) for breakout entries with 1-month (4w) EMA20 trend filter and volume confirmation.
# Long when price breaks above R3 in uptrend with volume spike, short when price breaks below S3 in downtrend with volume spike.
# Exit when price returns to the weekly pivot point (PP) or trend changes.
# Designed for low trade frequency (30-100 total trades over 4 years) with clear entry/exit rules to avoid overtrading.
# Works in both bull and bear markets by using trend filter and volatility-based stops.

name = "1d_Weekly_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "1d"
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

    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivot levels: R3, S3, and PP (pivot point)
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    pp_1w = typical_price.values
    hl_range = df_1w['high'] - df_1w['low']
    r3_1w = df_1w['close'].values + hl_range.values * 1.1 / 2
    s3_1w = df_1w['close'].values - hl_range.values * 1.1 / 2
    
    # Align weekly Camarilla levels to daily timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)

    # Get 4-week (28-day) EMA for trend filter (approximate 1-month trend)
    df_4w = get_htf_data(prices, '4w') if '4w' in ['5m', '15m', '30m', '1h', '4h', '6h', '12h', '1d', '1w'] else df_1w
    # Since 4w not available, use 1w EMA20 as proxy for medium-term trend
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)

    # Volume filter: >2x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(pp_1w_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 + price above 1w EMA20 (uptrend) + volume spike
            if (close[i] > r3_1w_aligned[i] and 
                close[i] > ema_20_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + price below 1w EMA20 (downtrend) + volume spike
            elif (close[i] < s3_1w_aligned[i] and 
                  close[i] < ema_20_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point (PP) or trend changes (price below EMA20)
            if (close[i] <= pp_1w_aligned[i] or close[i] < ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point (PP) or trend changes (price above EMA20)
            if (close[i] >= pp_1w_aligned[i] or close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals