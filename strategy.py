#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla pivot levels from 1d (R3/S3) with 1d EMA34 trend filter and volume spike confirmation capture mean reversion in range and breakout in trending markets. Works in bull/bear by fading extremes in range and following breakouts in trend.
Target: 25-40 trades/year per symbol with disciplined risk management.
"""

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
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

    # Get 1d data for Camarilla pivot and trend filter (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for current day (based on previous day)
    # Camarilla: Range = previous day's high - low
    # R3 = previous close + (Range * 1.1/2)
    # S3 = previous close - (Range * 1.1/2)
    # R4 = previous close + (Range * 1.1)
    # S4 = previous close - (Range * 1.1)
    range_1d = prev_high - prev_low
    camarilla_r3 = prev_close + (range_1d * 1.1 / 2)
    camarilla_s3 = prev_close - (range_1d * 1.1 / 2)
    camarilla_r4 = prev_close + (range_1d * 1.1)
    camarilla_s4 = prev_close - (range_1d * 1.1)
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels for current day)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.5x 24-period average (4 days of 6h data)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_24[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG ENTRY: Close below S3 (oversold) + 1d uptrend + volume spike -> mean reversion long
            # OR Close breaks above R4 (strong breakout) + 1d uptrend + volume spike -> continuation long
            if ((close[i] < camarilla_s3_aligned[i] and close[i] > ema34_1d_aligned[i]) or \
                (close[i] > camarilla_r4_aligned[i] and close[i] > ema34_1d_aligned[i])) and \
                volume[i] > vol_avg_24[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT ENTRY: Close above R3 (overbought) + 1d downtrend + volume spike -> mean reversion short
            # OR Close breaks below S4 (strong breakdown) + 1d downtrend + volume spike -> continuation short
            elif ((close[i] > camarilla_r3_aligned[i] and close[i] < ema34_1d_aligned[i]) or \
                  (close[i] < camarilla_s4_aligned[i] and close[i] < ema34_1d_aligned[i])) and \
                  volume[i] > vol_avg_24[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses above R3 (overbought) or below S3 (stop) or 1d trend turns down
            if close[i] > camarilla_r3_aligned[i] or close[i] < camarilla_s3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses below S3 (oversold) or above R3 (stop) or 1d trend turns up
            if close[i] < camarilla_s3_aligned[i] or close[i] > camarilla_r3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals