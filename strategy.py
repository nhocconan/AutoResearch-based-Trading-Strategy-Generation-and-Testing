#!/usr/bin/env python3
# 6h_Pivot_Reversal_1dTrend_Volume
# Hypothesis: Daily pivot point reversal on 6h timeframe with 1d trend filter and volume confirmation.
# Uses previous day's pivot (PP), resistance (R1), and support (S1) levels.
# Long when price crosses above PP in uptrend (close > EMA50) with volume spike.
# Short when price crosses below PP in downtrend (close < EMA50) with volume spike.
# Works in bull markets (buying dips above PP) and bear markets (selling rallies below PP).
# Target: 12-37 trades/year on 6h timeframe.

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
    
    # Get daily data for pivot point calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points using previous day's OHLC
    # PP = (High + Low + Close) / 3
    # R1 = 2*PP - Low
    # S1 = 2*PP - High
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid NaN from shift
    valid_idx = ~np.isnan(prev_high) & ~np.isnan(prev_low) & ~np.isnan(prev_close)
    pivot_point = np.full_like(prev_close, np.nan)
    r1 = np.full_like(prev_close, np.nan)
    s1 = np.full_like(prev_close, np.nan)
    
    pivot_point[valid_idx] = (prev_high[valid_idx] + prev_low[valid_idx] + prev_close[valid_idx]) / 3
    r1[valid_idx] = 2 * pivot_point[valid_idx] - prev_low[valid_idx]
    s1[valid_idx] = 2 * pivot_point[valid_idx] - prev_high[valid_idx]
    
    # Align pivot levels to 6h timeframe
    pivot_pp_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    pivot_r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    pivot_s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Price crosses above daily PP with volume in uptrend
            if (pivot_pp_aligned[i] > 0 and not np.isnan(pivot_pp_aligned[i]) and
                close[i] > pivot_pp_aligned[i] and
                low[i] <= pivot_pp_aligned[i] and  # crossed from below
                volume_confirmed[i] and
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below daily PP with volume in downtrend
            elif (pivot_pp_aligned[i] > 0 and not np.isnan(pivot_pp_aligned[i]) and
                  close[i] < pivot_pp_aligned[i] and
                  high[i] >= pivot_pp_aligned[i] and  # crossed from above
                  volume_confirmed[i] and
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S1 or trend weakens
            if (pivot_s1_aligned[i] > 0 and not np.isnan(pivot_s1_aligned[i]) and
                low[i] < pivot_s1_aligned[i]) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R1 or trend weakens
            if (pivot_r1_aligned[i] > 0 and not np.isnan(pivot_r1_aligned[i]) and
                high[i] > pivot_r1_aligned[i]) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals