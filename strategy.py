# 1d_PivotPoint_Bounce_1wTrend_Volume
# Hypothesis: On daily chart, price tends to bounce from pivot points (S1/S2/R1/R2) when aligned with weekly trend.
# Weekly EMA50 determines trend direction. Look for bounces from S1/S2 in uptrend or R1/R2 in downtrend.
# Volume confirmation ensures institutional participation. Low trade frequency (~10-20/year) avoids fee drag.
# Works in bull/bear: In uptrend, buy dips to support; in downtrend, sell rallies to resistance.

#!/usr/bin/env python3
name = "1d_PivotPoint_Bounce_1wTrend_Volume"
timeframe = "1d"
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
    
    # Weekly trend: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily pivot points (using prior day's data)
    # Shift by 1 to use only completed daily bars for pivot calculation
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = np.nan
    low_shift[0] = np.nan
    close_shift[0] = np.nan
    
    pivot = (high_shift + low_shift + close_shift) / 3
    range_hl = high_shift - low_shift
    
    # Support and resistance levels
    s1 = (2 * pivot) - high_shift
    s2 = pivot - range_hl
    r1 = (2 * pivot) - low_shift
    r2 = pivot + range_hl
    
    # Align pivot levels (they are already based on prior day, so no additional delay needed)
    s1_aligned = align_htf_to_ltf(prices, prices, s1)  # same timeframe, no shift
    s2_aligned = align_htf_to_ltf(prices, prices, s2)
    r1_aligned = align_htf_to_ltf(prices, prices, r1)
    r2_aligned = align_htf_to_ltf(prices, prices, r2)
    
    # Volume spike: current volume > 2.0 x 20-day average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(volume_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long setup: price near support in uptrend
            near_s1 = low[i] <= s1_aligned[i] * 1.002 and low[i] >= s1_aligned[i] * 0.998
            near_s2 = low[i] <= s2_aligned[i] * 1.002 and low[i] >= s2_aligned[i] * 0.998
            in_uptrend = close[i] > ema_50_1w_aligned[i]
            
            if (near_s1 or near_s2) and in_uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short setup: price near resistance in downtrend
            elif (high[i] >= r1_aligned[i] * 0.998 and high[i] <= r1_aligned[i] * 1.002) or \
                 (high[i] >= r2_aligned[i] * 0.998 and high[i] <= r2_aligned[i] * 1.002):
                near_r1 = high[i] >= r1_aligned[i] * 0.998 and high[i] <= r1_aligned[i] * 1.002
                near_r2 = high[i] >= r2_aligned[i] * 0.998 and high[i] <= r2_aligned[i] * 1.002
                in_downtrend = close[i] < ema_50_1w_aligned[i]
                if (near_r1 or near_r2) and in_downtrend and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price reaches midpoint or breaks support
            if high[i] >= pivot[i] * 0.998 or low[i] < s1_aligned[i] * 0.998:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches midpoint or breaks resistance
            if low[i] <= pivot[i] * 1.002 or high[i] > r1_aligned[i] * 1.002:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals