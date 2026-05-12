#!/usr/bin/env python3
"""
4h_TRIX_Volume_Spike_Chop_Filter
Hypothesis: TRIX (12-period triple EMA) with zero-crossovers captures momentum shifts.
Long when TRIX crosses above zero with volume > 1.5x 20-period average and choppy market (CHOP > 61.8).
Short when TRIX crosses below zero with volume surge and choppy market.
Uses 1d ADX < 25 as additional range filter to avoid trending markets where TRIX whipsaws.
Designed for 4h timeframe to target 20-50 trades/year with low turnover.
"""

name = "4h_TRIX_Volume_Spike_Chop_Filter"
timeframe = "4h"
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

    # Calculate TRIX: triple EMA of close, then percent change
    def ema(series, period):
        return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

    ema1 = ema(close, 12)
    ema2 = ema(ema1, 12)
    ema3 = ema(ema2, 12)
    # TRIX = 100 * (EMA3 - previous EMA3) / previous EMA3
    trix_raw = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix_raw[0] = 0  # first value has no previous

    # Get daily data for ADX and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate ADX(14) for trend strength
    def wilders_smoothing(series, period):
        result = np.full_like(series, np.nan)
        if len(series) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(series[:period])
        for i in range(period, len(series)):
            result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result

    # TR calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    # Add leading NaN for alignment
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])

    # Smooth TR, +DM, -DM
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)

    # Calculate Chopiness Index(14)
    def chop_index(high, low, close, period):
        # True Range sum
        tr_sum = np.nancumsum(tr) - np.concatenate([[0], np.nancumsum(tr)[:-period]]) if len(tr) >= period else np.full_like(tr, np.nan)
        # Highest high - lowest low over period
        hh = np.full_like(high, np.nan)
        ll = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        # Avoid division by zero
        hh_ll = hh - ll
        chop = 100 * np.log10(tr_sum / hh_ll) / np.log10(period)
        return chop

    chop = chop_index(high_1d, low_1d, close_1d, 14)

    # Align HTF indicators
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)

    # Volume confirmation: 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Get aligned values for current 4h bar
        adx_val = adx_aligned[i]
        chop_val = chop_aligned[i]
        trix = trix_raw[i]
        trix_prev = trix_raw[i-1]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(adx_val) or np.isnan(chop_val) or 
            np.isnan(trix) or np.isnan(trix_prev) or 
            np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Range filter: only trade when ADX < 25 (weak trend) and CHOP > 61.8 (choppy)
        if adx_val >= 25 or chop_val <= 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + volume surge
            if (trix > 0 and trix_prev <= 0 and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + volume surge
            elif (trix < 0 and trix_prev >= 0 and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero
            if trix < 0 and trix_prev >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero
            if trix > 0 and trix_prev <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals