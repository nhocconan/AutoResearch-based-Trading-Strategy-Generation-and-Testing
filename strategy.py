#!/usr/bin/env python3
# 1D_1W_Camarilla_R3S3_Breakout_Volume_Trend
# Hypothesis: Daily price breaks at weekly-derived Camarilla R3/S3 levels with volume confirmation
# and weekly trend filter (price above/below weekly EMA50) capture high-probability moves.
# Weekly trend alignment reduces whipsaws in both bull and bear markets. Targets low-frequency,
# high-quality setups (10-25 trades/year) to minimize fee drag.

name = "1D_1W_Camarilla_R3S3_Breakout_Volume_Trend"
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

    # Get weekly data for trend filter and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')

    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Calculate daily Camarilla levels: R3, S3
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    rng = high_1d - low_1d
    r3 = close_1d + 1.1 * rng
    s3 = close_1d - 1.1 * rng
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)

    # Volume spike: volume > 2.0 * 20-period average (~20 days at 1d)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above weekly EMA50 (uptrend) + breaks above R3 + volume spike
            if close[i] > ema50_1w_aligned[i] and close[i] > r3_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below weekly EMA50 (downtrend) + breaks below S3 + volume spike
            elif close[i] < ema50_1w_aligned[i] and close[i] < s3_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 (reversal) or weekly trend turns bearish
            if close[i] < s3_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 (reversal) or weekly trend turns bullish
            if close[i] > r3_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals