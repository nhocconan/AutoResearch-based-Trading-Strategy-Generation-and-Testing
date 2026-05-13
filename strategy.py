#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Breakout_Trend
Hypothesis: Combines weekly pivot levels (from 1w data) with 1d trend filter and volume confirmation.
Breakouts above weekly R1 or below weekly S1 with trend alignment capture strong moves.
Designed for low trade frequency (15-25/year) to work in both bull and bear markets by following
the dominant weekly trend while using volatility filters to avoid chop.
"""

name = "6h_Weekly_Pivot_Breakout_Trend"
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
    
    # Calculate weekly pivot points (using prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    # Prior week's OHLC (shift by 1 to avoid look-ahead)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    
    # Calculate pivot point and support/resistance levels
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = 2 * pivot - prev_week_low
    s1 = 2 * pivot - prev_week_high
    r2 = pivot + (prev_week_high - prev_week_low)
    s2 = pivot - (prev_week_high - prev_week_low)
    
    # Align weekly levels to 6h timeframe (already delayed by shift(1))
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2.values)
    
    # 1-day trend filter: EMA 34
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Volatility filter: avoid low volatility periods (ATR ratio)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr > (0.5 * atr_ma)  # Avoid extremely low volatility
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Break above weekly R1 with 1d uptrend and volume/volatility confirmation
            if (close[i] > r1_aligned[i] and 
                ema_34_1d_aligned[i] > 0 and 
                close[i] > ema_34_1d_aligned[i] and
                volume_confirm[i] and
                volatility_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly S1 with 1d downtrend and volume/volatility confirmation
            elif (close[i] < s1_aligned[i] and 
                  ema_34_1d_aligned[i] > 0 and 
                  close[i] < ema_34_1d_aligned[i] and
                  volume_confirm[i] and
                  volatility_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly pivot or 1d trend turns down
            if (close[i] < pivot_aligned[i]) or \
               (ema_34_1d_aligned[i] > 0 and close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly pivot or 1d trend turns up
            if (close[i] > pivot_aligned[i]) or \
               (ema_34_1d_aligned[i] > 0 and close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals