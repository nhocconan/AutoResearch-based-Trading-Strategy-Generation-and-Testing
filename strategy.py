#!/usr/bin/env python3
# 4h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Use 1d Camarilla pivot levels (R3/S3) for breakout entries with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above R3 in uptrend with volume spike, short when price breaks below S3 in downtrend with volume spike.
# Exit when price returns to the 1d pivot level (PP) or trend changes.
# Designed for low trade frequency (20-50 total trades over 4 years) with clear entry/exit rules to avoid overtrading.

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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

    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels: R3, S3, and PP (pivot point)
    # Camarilla formulas:
    # PP = (H + L + C) / 3
    # R3 = C + (H - L) * 1.1 / 2
    # S3 = C - (H - L) * 1.1 / 2
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    pp_1d = typical_price.values
    hl_range = df_1d['high'] - df_1d['low']
    r3_1d = df_1d['close'].values + hl_range.values * 1.1 / 2
    s3_1d = df_1d['close'].values - hl_range.values * 1.1 / 2
    
    # Align 1d Camarilla levels to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)

    # Get 4h data for EMA trend filter (same timeframe)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(pp_1d_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 + price above 4h EMA50 (uptrend) + volume spike
            if (close[i] > r3_1d_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + price below 4h EMA50 (downtrend) + volume spike
            elif (close[i] < s3_1d_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point (PP) or trend changes (price below EMA50)
            if (close[i] <= pp_1d_aligned[i] or close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point (PP) or trend changes (price above EMA50)
            if (close[i] >= pp_1d_aligned[i] or close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals