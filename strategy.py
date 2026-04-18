#!/usr/bin/env python3
"""
6h_1d_Pivot_R3S3_Fade_With_Volume_Confirmation
Hypothesis: On 6h timeframe, fade at daily Pivot R3/S3 levels during ranging markets, but allow breakout continuation at R4/S4 when accompanied by volume spikes. This strategy aims to profit from mean reversion in sideways markets while capturing strong directional moves when momentum builds. Uses volume confirmation to reduce false signals and targets 15-35 trades per year to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points and support/resistance levels
    # Pivot = (High + Low + Close) / 3
    # R1 = 2*Pivot - Low
    # S1 = 2*Pivot - High
    # R2 = Pivot + (High - Low)
    # S2 = Pivot - (High - Low)
    # R3 = High + 2*(Pivot - Low)
    # S3 = Low - 2*(High - Pivot)
    # R4 = R3 + (High - Low)
    # S4 = S3 - (High - Low)
    
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_daily = prev_high - prev_low
    
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + range_daily
    s2 = pivot - range_daily
    r3 = prev_high + 2 * (pivot - prev_low)
    s3 = prev_low - 2 * (prev_high - pivot)
    r4 = r3 + range_daily
    s4 = s3 - range_daily
    
    # Align daily pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: current volume > 1.8x 24-period average (48h for 6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > 1.8 * vol_ma
    vol_confirm = np.where(np.isnan(vol_confirm), False, vol_confirm)
    
    # Choppiness filter: avoid trading in extremely choppy conditions
    # Calculate Choppy Index using ATR and price range over 14 periods
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(np.roll(close, 1) - low)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    
    atr_period = 14
    atr = np.zeros_like(tr)
    atr[atr_period] = np.nansum(tr[1:atr_period+1]) if not np.isnan(tr).all() else 0
    for i in range(atr_period + 1, len(tr)):
        atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Calculate highest high and lowest low over atr_period
    highest_high = np.zeros_like(high)
    lowest_low = np.zeros_like(low)
    for i in range(len(high)):
        if i < atr_period:
            highest_high[i] = np.max(high[:i+1]) if i >= 0 else high[i]
            lowest_low[i] = np.min(low[:i+1]) if i >= 0 else low[i]
        else:
            highest_high[i] = np.max(high[i-atr_period+1:i+1])
            lowest_low[i] = np.min(low[i-atr_period+1:i+1])
    
    # Choppy Index: 100 * log10(sum(TR, atr_period) / (atr_period * (highest_high - lowest_low))) / log10(atr_period)
    tr_sum = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < atr_period:
            tr_sum[i] = np.sum(tr[:i+1]) if i >= 0 else tr[i]
        else:
            tr_sum[i] = np.sum(tr[i-atr_period+1:i+1])
    
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    
    chop = np.zeros_like(tr)
    for i in range(len(tr)):
        if hl_range[i] > 0 and tr_sum[i] > 0:
            chop[i] = 100 * np.log10(tr_sum[i] / (atr_period * hl_range[i])) / np.log10(atr_period)
        else:
            chop[i] = 50  # neutral value
    
    # Market regime: chop > 61.8 = ranging (mean revert), chop < 38.2 = trending (breakout)
    chop_above = chop > 61.8
    chop_below = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need enough for ATR and Chop calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long conditions:
            # 1. In ranging market (chop > 61.8) and price at S3 with volume confirmation -> mean reversion long
            # 2. In trending market (chop < 38.2) and price breaks above R4 with volume -> breakout long
            if ((chop_above[i] and close[i] <= s3_aligned[i] and vol_confirm[i]) or
                (chop_below[i] and close[i] >= r4_aligned[i] and vol_confirm[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions:
            # 1. In ranging market (chop > 61.8) and price at R3 with volume confirmation -> mean reversion short
            # 2. In trending market (chop < 38.2) and price breaks below S4 with volume -> breakout short
            elif ((chop_above[i] and close[i] >= r3_aligned[i] and vol_confirm[i]) or
                  (chop_below[i] and close[i] <= s4_aligned[i] and vol_confirm[i])):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit conditions:
            # 1. Price reaches R3 in ranging market (take profit at opposite level)
            # 2. Price falls below S4 in trending market (stop loss)
            # 3. Volume drops significantly (loss of momentum)
            if ((chop_above[i] and close[i] >= r3_aligned[i]) or
                (chop_below[i] and close[i] < s4_aligned[i]) or
                (volume[i] < 0.5 * vol_ma[i])):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions:
            # 1. Price reaches S3 in ranging market (take profit at opposite level)
            # 2. Price rises above R4 in trending market (stop loss)
            # 3. Volume drops significantly (loss of momentum)
            if ((chop_above[i] and close[i] <= s3_aligned[i]) or
                (chop_below[i] and close[i] > r4_aligned[i]) or
                (volume[i] < 0.5 * vol_ma[i])):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Pivot_R3S3_Fade_With_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0