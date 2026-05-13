# 12h_1W_1D_Camarilla_R3_S3_Breakout_With_Trend_Filter
# Hypothesis: Strong moves occur when price breaks weekly/daily Camarilla R3/S3 levels
# with volume confirmation and aligned with weekly trend. Uses weekly EMA50 as trend filter
# to avoid counter-trend trades. Designed for low-frequency, high-quality setups (12-37 trades/year)
# to minimize fee drag and work in both bull and bear markets by following higher timeframe momentum.

name = "12h_1W_1D_Camarilla_R3_S3_Breakout_With_Trend_Filter"
timeframe = "12h"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')

    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Calculate daily Camarilla levels: R3, S3
    # Camarilla: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    rng = high_1d - low_1d
    r3 = close_1d + 1.1 * rng
    s3 = close_1d - 1.1 * rng
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)

    # Volume spike: volume > 2.0 * 24-period average (~12 days at 12h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma_24

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