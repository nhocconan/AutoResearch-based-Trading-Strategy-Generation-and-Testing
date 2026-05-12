#!/usr/bin/env python3
# 6h_1D_Camarilla_R3_S3_Fade_1dTrend_VolumeFilter_v1
# Hypothesis: Fade at Camarilla R3/S3 levels from 1d timeframe when price shows rejection (close opposite direction of wick) in ranging markets (ADX < 25 on 1d), with volume confirmation to avoid false signals. In trending markets (ADX >= 25), follow breakouts at R4/S4 levels. Uses 1d trend filter (EMA50) to avoid counter-trend trades. Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drift. Works in bull/bear by adapting to market regime.

name = "6h_1D_Camarilla_R3_S3_Fade_1dTrend_VolumeFilter_v1"
timeframe = "6h"
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

    # Get 1d data for Camarilla levels, trend, ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate 1d Camarilla levels (based on previous day)
    # R4 = Close + 1.5*(High-Low), R3 = Close + 1.0*(High-Low), etc.
    # But we use typical formula based on previous day's range
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First value will be NaN due to roll, handled by min_periods later
    range_1d = prev_high - prev_low
    camarilla_r3 = prev_close + 1.0 * range_1d
    camarilla_s3 = prev_close - 1.0 * range_1d
    camarilla_r4 = prev_close + 1.5 * range_1d
    camarilla_s4 = prev_close - 1.5 * range_1d

    # Align Camarilla levels to 6h timeframe (using close_1d for alignment index)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)

    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate 1d ADX for regime detection (trend strength)
    # ADX calculation: +DM, -DM, TR, then smoothed
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # First TR will use rolled value, but we'll handle with min_periods in smoothing
    atr_period = 14
    atr = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False).mean().values
    # Smooth +DM and -DM
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/atr_period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/atr_period, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/atr_period, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)

    # Calculate 6h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Determine market regime using 1d ADX
            if adx_aligned[i] < 25:  # Ranging market: fade at R3/S3
                # Check for rejection at R3: long wick above, close near low
                r3_rejection = (high[i] > camarilla_r3_aligned[i]) and (close[i] < (camarilla_r3_aligned[i] + (high[i] - camarilla_r3_aligned[i]) * 0.3))
                # Check for rejection at S3: long wick below, close near high
                s3_rejection = (low[i] < camarilla_s3_aligned[i]) and (close[i] > (camarilla_s3_aligned[i] + (camarilla_s3_aligned[i] - low[i]) * 0.3))
                
                if r3_rejection and volume[i] > volume_sma20[i]:
                    # Fade R3: expect price to go down from resistance
                    signals[i] = -0.25
                    position = -1
                elif s3_rejection and volume[i] > volume_sma20[i]:
                    # Fade S3: expect price to go up from support
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            else:  # Trending market: breakout at R4/S4
                # Breakout above R4 in uptrend
                if close[i] > camarilla_r4_aligned[i] and close[i] > ema50_1d_aligned[i] and volume[i] > volume_sma20[i]:
                    signals[i] = 0.25
                    position = 1
                # Breakdown below S4 in downtrend
                elif close[i] < camarilla_s4_aligned[i] and close[i] < ema50_1d_aligned[i] and volume[i] > volume_sma20[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 1d EMA50 (trend change) or hits S3 in ranging market
            if adx_aligned[i] < 25:  # Ranging: take profit at S3
                if close[i] < camarilla_s3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Trending: follow trend until EMA50 break
                if close[i] < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 1d EMA50 (trend change) or hits R3 in ranging market
            if adx_aligned[i] < 25:  # Ranging: take profit at R3
                if close[i] > camarilla_r3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Trending: follow trend until EMA50 break
                if close[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25

    return signals