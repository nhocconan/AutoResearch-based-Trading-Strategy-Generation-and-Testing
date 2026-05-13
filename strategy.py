#!/usr/bin/env python3
# 6h_Pivot_Reversal_1dTrend_Volume
# Hypothesis: Daily pivot reversals on 6h chart with 1d trend filter (EMA34) and volume confirmation.
# Works in bull markets (long at daily S1/S2 in uptrend) and bear markets (short at daily R1/R2 in downtrend).
# Target: 15-30 trades/year to minimize fee drag on 6h timeframe.

name = "6h_Pivot_Reversal_1dTrend_Volume"
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
    
    # Get 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivots using previous day's OHLC
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid NaN from shift
    valid_idx = ~np.isnan(prev_high) & ~np.isnan(prev_low) & ~np.isnan(prev_close)
    pivot = np.full_like(prev_close, np.nan)
    r1 = np.full_like(prev_close, np.nan)
    s1 = np.full_like(prev_close, np.nan)
    r2 = np.full_like(prev_close, np.nan)
    s2 = np.full_like(prev_close, np.nan)
    
    pivot[valid_idx] = (prev_high[valid_idx] + prev_low[valid_idx] + prev_close[valid_idx]) / 3
    r1[valid_idx] = 2 * pivot[valid_idx] - prev_low[valid_idx]
    s1[valid_idx] = 2 * pivot[valid_idx] - prev_high[valid_idx]
    r2[valid_idx] = pivot[valid_idx] + (prev_high[valid_idx] - prev_low[valid_idx])
    s2[valid_idx] = pivot[valid_idx] - (prev_high[valid_idx] - prev_low[valid_idx])
    
    # Align pivots to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Get 1d data for EMA trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Price reverses up from S1 or S2 with volume confirmation in uptrend
            if ((s1_aligned[i] > 0 and not np.isnan(s1_aligned[i]) and low[i] <= s1_aligned[i]) or
                (s2_aligned[i] > 0 and not np.isnan(s2_aligned[i]) and low[i] <= s2_aligned[i])) and \
               volume_confirmed[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price reverses down from R1 or R2 with volume confirmation in downtrend
            elif ((r1_aligned[i] > 0 and not np.isnan(r1_aligned[i]) and high[i] >= r1_aligned[i]) or
                  (r2_aligned[i] > 0 and not np.isnan(r2_aligned[i]) and high[i] >= r2_aligned[i])) and \
                 volume_confirmed[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches R1 or trend weakens
            if ((r1_aligned[i] > 0 and not np.isnan(r1_aligned[i]) and high[i] >= r1_aligned[i]) or
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches S1 or trend weakens
            if ((s1_aligned[i] > 0 and not np.isnan(s1_aligned[i]) and low[i] <= s1_aligned[i]) or
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals