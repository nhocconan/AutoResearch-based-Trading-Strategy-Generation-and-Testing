#!/usr/bin/env python3
"""
6h_Wilson_Wave_Trader
Hypothesis: Combines Wilson Wave (modified RSI divergence) with 1w pivot structure and volume confirmation to capture trend reversals in both bull and bear markets.
Wilson Wave identifies exhaustion via RSI divergence, weekly pivots provide institutional support/resistance, and volume confirms institutional participation.
Target: 15-25 trades/year per symbol with controlled risk in all market regimes.
"""

name = "6h_Wilson_Wave_Trader"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def wilson_wave(rsi_series, period=14):
    """Calculate Wilson Wave oscillator from RSI"""
    # Wilson Wave = RSI - EMA(RSI)
    rsi_ema = pd.Series(rsi_series).ewm(span=period, adjust=False).mean()
    wilson = rsi_series - rsi_ema.values
    return wilson

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for pivot points (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    pivot = typical_price.values
    r1 = 2 * pivot - df_1w['low'].values
    s1 = 2 * pivot - df_1w['high'].values
    r2 = pivot + (df_1w['high'].values - df_1w['low'].values)
    s2 = pivot - (df_1w['high'].values - df_1w['low'].values)
    
    # Align weekly pivots to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)

    # Calculate RSI (14-period) for Wilson Wave
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Wilson Wave
    wilson = wilson_wave(rsi, 14)
    
    # Volume confirmation: volume > 1.5x 24-period average (4 days)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required data is NaN
        if (np.isnan(wilson[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Wilson Wave bullish divergence + price above S1 + volume confirmation
            if (wilson[i] > wilson[i-1] and  # Wilson turning up
                close[i] > s1_aligned[i] and  # Above weekly S1
                volume[i] > vol_avg_24[i] * 1.5):  # Volume spike
                signals[i] = 0.25
                position = 1
            # SHORT: Wilson Wave bearish divergence + price below R1 + volume confirmation
            elif (wilson[i] < wilson[i-1] and  # Wilson turning down
                  close[i] < r1_aligned[i] and  # Below weekly R1
                  volume[i] > vol_avg_24[i] * 1.5):  # Volume spike
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Wilson Wave turns down OR price breaks below pivot
            if wilson[i] < wilson[i-1] or close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Wilson Wave turns up OR price breaks above pivot
            if wilson[i] > wilson[i-1] or close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals