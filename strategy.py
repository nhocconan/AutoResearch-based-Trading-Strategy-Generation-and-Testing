#!/usr/bin/env python3
# 6h_ADX_Supertrend_Momentum
# Hypothesis: Combine ADX trend strength with Supertrend momentum and volume confirmation.
# ADX > 25 filters for trending markets, Supertrend gives direction, volume confirms momentum.
# Works in bull/bear by only taking strong trend moves. Target: 20-30 trades/year.

name = "6h_ADX_Supertrend_Momentum"
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

    # Get 1d data for ADX and Supertrend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate ATR for Supertrend
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values

    # Supertrend calculation
    upper_band = (high_1d + low_1d) / 2 + 3 * atr
    lower_band = (high_1d + low_1d) / 2 - 3 * atr
    supertrend = np.zeros_like(close_1d)
    supertrend_direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend

    for i in range(1, len(close_1d)):
        if close_1d[i] > upper_band[i-1]:
            supertrend_direction[i] = 1
        elif close_1d[i] < lower_band[i-1]:
            supertrend_direction[i] = -1
        else:
            supertrend_direction[i] = supertrend_direction[i-1]
            if supertrend_direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if supertrend_direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]

        supertrend[i] = lower_band[i] if supertrend_direction[i] == 1 else upper_band[i]

    # Calculate ADX
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)

    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values

    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    # Align HTF indicators to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_1d, supertrend_direction)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):  # Start after ADX warmup
        # Get aligned values
        st = supertrend_aligned[i]
        std = supertrend_dir_aligned[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike_aligned[i]

        if np.isnan(st) or np.isnan(std) or np.isnan(adx_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend (Supertrend direction = 1) + Strong trend (ADX > 25) + Volume spike
            if std == 1 and adx_val > 25 and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend (Supertrend direction = -1) + Strong trend (ADX > 25) + Volume spike
            elif std == -1 and adx_val > 25 and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend weakness (ADX < 20) or Supertrend flip
            if adx_val < 20 or std == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend weakness (ADX < 20) or Supertrend flip
            if adx_val < 20 or std == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals