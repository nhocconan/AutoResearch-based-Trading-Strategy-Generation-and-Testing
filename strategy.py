#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Breakout above Camarilla R3 or below S3 on 12h with 1d trend filter (EMA34) and volume spike.
Works in bull/bear by following daily trend, avoids whipsaws via volatility filter.
Targets 15-30 trades/year.
"""

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_ = high - low
    if range_ == 0:
        return close, close, close, close, close, close
    c = close
    h = high
    l = low
    R4 = c + (range_ * 1.1 / 2)
    R3 = c + (range_ * 1.1/4)
    R2 = c + (range_ * 1.1/6)
    R1 = c + (range_ * 1.1/12)
    S1 = c - (range_ * 1.1/12)
    S2 = c - (range_ * 1.1/6)
    S3 = c - (range_ * 1.1/4)
    S4 = c - (range_ * 1.1/2)
    return R3, R2, R1, S1, S2, S3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get daily data for trend and Camarilla levels ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_prev = np.roll(ema_34, 1)
    ema_34_prev[0] = ema_34[0]
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize arrays for Camarilla levels
    R3_1d = np.full_like(close_1d, np.nan)
    S3_1d = np.full_like(close_1d, np.nan)
    
    # Calculate Camarilla for each day (using previous day's data)
    for i in range(1, len(close_1d)):
        R3, _, _, _, _, S3 = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
        R3_1d[i] = R3
        S3_1d[i] = S3
    
    # For first day, use same values (no previous day)
    if len(close_1d) > 0:
        R3, _, _, _, _, S3 = calculate_camarilla(high_1d[0], low_1d[0], close_1d[0])
        R3_1d[0] = R3
        S3_1d[0] = S3

    # Align to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    ema_34_prev_aligned = align_htf_to_ltf(prices, df_1d, ema_34_prev)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)

    # Volume spike: current > 2.0x average of last 4 periods
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after EMA warmup
        if (np.isnan(ema_34_aligned[i]) or np.isnan(ema_34_prev_aligned[i]) or 
            np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price > R3 + daily uptrend + volume spike
            if (close[i] > R3_1d_aligned[i] and 
                ema_34_aligned[i] > ema_34_prev_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price < S3 + daily downtrend + volume spike
            elif (close[i] < S3_1d_aligned[i] and 
                  ema_34_aligned[i] < ema_34_prev_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < S3 (reversal to downside)
            if close[i] < S3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > R3 (reversal to upside)
            if close[i] > R3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals