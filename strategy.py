#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Weekly_ADX_Filter
Hypothesis: Use daily Kaufman Adaptive Moving Average (KAMA) for trend direction, filtered by weekly ADX > 25 to ensure trending markets. Enter long when price crosses above KAMA, short when price crosses below KAMA. Exit on opposite cross. This avoids choppy markets and captures sustained trends in both bull and bear cycles.
Timeframe: 1d
"""

name = "1d_KAMA_Trend_With_Weekly_ADX_Filter"
timeframe = "1d"
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

    # Get weekly data for ADX filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)

    # Calculate weekly ADX (14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period

    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0

    # Smooth TR, DM+, DM- (Wilder smoothing = EMA with alpha=1/period)
    def wilders_smooth(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result

    atr = wilders_smooth(tr, 14)
    dm_plus_smooth = wilders_smooth(dm_plus, 14)
    dm_minus_smooth = wilders_smooth(dm_minus, 14)

    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)

    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smooth(dx, 14)
    adx_14 = adx  # ADX(14)

    # Align weekly ADX to daily
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_14)

    # Daily KAMA (10, 2, 30)
    def kama(close, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=10))
        volatility = np.nansum(np.abs(np.diff(close)), axis=0) if len(close) >= 10 else np.full_like(change, np.nan)
        # Actually compute properly
        er = np.full_like(close, np.nan)
        for i in range(10, len(close)):
            if np.isnan(close[i-10:i]).any() or np.isnan(close[i-10+1:i+1]).any():
                er[i] = np.nan
            else:
                change_val = np.abs(close[i] - close[i-10])
                volatility_val = np.nansum(np.abs(np.diff(close[i-10:i+1])))
                er[i] = change_val / volatility_val if volatility_val != 0 else 0
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama_val = np.full_like(close, np.nan)
        kama_val[0] = close[0]
        for i in range(1, len(close)):
            if np.isnan(sc[i]):
                kama_val[i] = kama_val[i-1]
            else:
                kama_val[i] = kama_val[i-1] + sc[i] * (close[i] - kama_val[i-1])
        return kama_val

    kama_val = kama(close, 2, 30)
    # Warmup period: need at least 30 for KAMA stability
    warmup = 30

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(warmup, n):
        if np.isnan(kama_val[i]) or np.isnan(adx_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Only trade when weekly ADX > 25 (trending market)
        if adx_aligned[i] > 25:
            if position == 0:
                if close[i] > kama_val[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < kama_val[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                if close[i] < kama_val[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if close[i] > kama_val[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In choppy market (ADX <= 25), stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0

    return signals