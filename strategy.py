# 12h_PriceAction_Reversal_at_DailyPivots_VolumeFilter
# Hypothesis: Price reverses at daily pivot levels (support/resistance) with volume confirmation.
# Works in bull/bear: buys at support, sells at resistance. Uses volume to confirm institutional interest.
# Timeframe: 12h (lower trade frequency, less fee drag). HTF: 1d for pivot calculation.
# Expected trades: ~15-25/year per symbol (within 50-150 total over 4 years target).
# Uses: Daily Pivot Points (classic formula), 12h price action, volume filter.
# Risk: Exit on opposite pivot touch or trend change via EMA34.

#!/usr/bin/env python3
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
    
    # Get daily data for pivot points (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points: P = (H+L+C)/3, S1 = 2P-H, R1 = 2P-L, S2 = P-(H-L), R2 = P+(H-L)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    s1 = 2 * pivot - high_1d
    r1 = 2 * pivot - low_1d
    s2 = pivot - (high_1d - low_1d)
    r2 = pivot + (high_1d - low_1d)
    
    # Align daily pivot levels to 12h timeframe (wait for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    
    # Volume filter: volume > 1.5x 20-period average (moderate to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Trend filter: 12h EMA34 to avoid trading against strong trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 35:
        return np.zeros(n)
    close_12h = pd.Series(df_12h['close'].values)
    ema34_12h = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s2_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(ema34_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long setup: price at or below S1/S2 with volume confirmation AND above EMA34 (not in strong downtrend)
            if ((low[i] <= s1_aligned[i] or low[i] <= s2_aligned[i]) and 
                volume_filter[i] and 
                close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: price at or above R1/R2 with volume confirmation AND below EMA34 (not in strong uptrend)
            elif ((high[i] >= r1_aligned[i] or high[i] >= r2_aligned[i]) and 
                  volume_filter[i] and 
                  close[i] < ema34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price reaches R1/R2 (profit target) or closes below EMA34 (trend change)
            if high[i] >= r1_aligned[i] or high[i] >= r2_aligned[i] or close[i] < ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches S1/S2 (profit target) or closes above EMA34 (trend change)
            if low[i] <= s1_aligned[i] or low[i] <= s2_aligned[i] or close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_PriceAction_Reversal_at_DailyPivots_VolumeFilter"
timeframe = "12h"
leverage = 1.0