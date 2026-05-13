#!/usr/bin/env python3
# 6h_Weekly_Pivot_Breakout_1dTrend_Volume
# Hypothesis: Use weekly pivot points (from weekly high/low/close) to identify key support/resistance.
# Long when price breaks above weekly R2 in uptrend (price > 1d EMA50) with volume confirmation.
# Short when price breaks below weekly S2 in downtrend (price < 1d EMA50) with volume confirmation.
# Exit when price returns to weekly pivot point (PP) or trend reverses.
# Weekly pivots provide structure that works in both trending and ranging markets.
# Target: 50-150 total trades over 4 years with disciplined risk control.

name = "6h_Weekly_Pivot_Breakout_1dTrend_Volume"
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

    # Get weekly data for pivot point calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: PP, R1, S1, R2, S2
    # Standard pivot point formulas:
    # PP = (H + L + C) / 3
    # R1 = (2 * PP) - L
    # S1 = (2 * PP) - H
    # R2 = PP + (H - L)
    # S2 = PP - (H - L)
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    pp_1w = typical_price.values
    hl_range = df_1w['high'] - df_1w['low']
    r1_1w = (2 * pp_1w) - df_1w['low'].values
    s1_1w = (2 * pp_1w) - df_1w['high'].values
    r2_1w = pp_1w + hl_range.values
    s2_1w = pp_1w - hl_range.values
    
    # Align weekly pivot levels to 6h timeframe
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or 
            np.isnan(s2_1w_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above weekly R2 + price above 1d EMA50 (uptrend) + volume spike
            if (close[i] > r2_1w_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly S2 + price below 1d EMA50 (downtrend) + volume spike
            elif (close[i] < s2_1w_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to weekly pivot point (PP) or trend changes (price below EMA50)
            if (close[i] <= pp_1w_aligned[i] or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to weekly pivot point (PP) or trend changes (price above EMA50)
            if (close[i] >= pp_1w_aligned[i] or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals