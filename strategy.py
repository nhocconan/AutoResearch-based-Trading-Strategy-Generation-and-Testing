#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeFilter
Hypothesis: On 1h timeframe, use Camarilla pivot levels (R3/S3) from prior 4h bar for breakouts, filtered by 4h EMA50 trend direction and volume >1.5x average to avoid false signals. Targets 20-50 trades/year by requiring confluence of Camarilla breakout, trend alignment, and volume confirmation. Works in bull/bear markets via trend filter.
"""

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 4h data for Camarilla pivots and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R3, S3) for each 4h bar
    # Formula: R3 = Close + (High - Low) * 1.1/2, S3 = Close - (High - Low) * 1.1/2
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla R3 and S3
    camarilla_r3 = close_4h + (high_4h - low_4h) * 1.1 / 2
    camarilla_s3 = close_4h - (high_4h - low_4h) * 1.1 / 2
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Align to 1h timeframe (wait for 4h bar close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)

    # Volume filter: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):  # Warmup for indicators
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above prior 4h R3 + 4h uptrend + volume spike
            if (close[i] > camarilla_r3_aligned[i-1] and 
                close[i] > ema50_4h_aligned[i] and 
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below prior 4h S3 + 4h downtrend + volume spike
            elif (close[i] < camarilla_s3_aligned[i-1] and 
                  close[i] < ema50_4h_aligned[i] and 
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below prior 4h S3 OR trend turns down
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above prior 4h R3 OR trend turns up
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals