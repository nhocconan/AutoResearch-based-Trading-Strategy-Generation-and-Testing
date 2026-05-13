#!/usr/bin/env python3
# 1d_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Use weekly Camarilla pivot levels (R3/S3) for breakout entries with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above weekly R3 in uptrend with volume spike, short when price breaks below weekly S3 in downtrend with volume spike.
# Exit when price returns to weekly pivot point (PP) or trend changes.
# Weekly pivots provide stronger support/resistance than daily, reducing false breakouts.
# Designed for low trade frequency (30-100 total trades over 4 years) to minimize fee drag.

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

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for Camarilla pivot point calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivot points: PP, R3, S3
    # Standard Camarilla formulas:
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1 / 4
    # S3 = PP - (H - L) * 1.1 / 4
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    pp_1w = (high_w + low_w + close_w) / 3
    r3_1w = pp_1w + (high_w - low_w) * 1.1 / 4
    s3_1w = pp_1w - (high_w - low_w) * 1.1 / 4
    
    # Align weekly Camarilla levels to 1d timeframe
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)

    # Get weekly data for EMA trend filter
    ema_50_1w = pd.Series(close_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Volume filter: >1.8x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(pp_1w_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 + price above weekly EMA50 (uptrend) + volume spike
            if (close[i] > r3_1w_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below S3 + price below weekly EMA50 (downtrend) + volume spike
            elif (close[i] < s3_1w_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.8):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point (PP) or trend changes (price below EMA50)
            if (close[i] <= pp_1w_aligned[i] or close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point (PP) or trend changes (price above EMA50)
            if (close[i] >= pp_1w_aligned[i] or close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals