#!/usr/bin/env python3
# 6H_PivotReversal_WithVolumeFilter
# Hypothesis: Fade extreme daily price movements using Camarilla pivot levels (R3/S3) with volume confirmation.
# In strong trends, price often reverts to the mean after reaching extreme levels.
# Uses 12h trend filter to avoid counter-trend trades during strong moves.
# Works in bull/bear by fading extremes regardless of direction.
# Target: 15-30 trades/year per symbol.

name = "6H_PivotReversal_WithVolumeFilter"
timeframe = "6h"
leverage = 1.0

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
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1)
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    R3 = prev_close + (range_ * 1.1 / 4)
    S3 = prev_close - (range_ * 1.1 / 4)
    R4 = prev_close + (range_ * 1.1 / 2)
    S4 = prev_close - (range_ * 1.1 / 2)
    
    # Align daily levels to 6h
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    
    # 12h trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Simple trend: price above/below previous close
    prev_close_12h = np.concatenate([[np.nan], close_12h[:-1]])
    trend_up_12h = close_12h > prev_close_12h
    trend_down_12h = close_12h < prev_close_12h
    
    # Align 12h trend to 6h
    trend_up_12h_6h = align_htf_to_ltf(prices, df_12h, trend_up_12h.astype(float))
    trend_down_12h_6h = align_htf_to_ltf(prices, df_12h, trend_down_12h.astype(float))
    
    # Volume filter (20-period average)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or np.isnan(R4_6h[i]) or np.isnan(S4_6h[i]) or
            np.isnan(trend_up_12h_6h[i]) or np.isnan(trend_down_12h_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: price at S3 (strong support) with volume, no strong 12h downtrend
            if (close[i] <= S3_6h[i] * 1.005 and  # Near S3
                close[i] >= S3_6h[i] * 0.995 and
                volume_confirm and
                not trend_down_12h_6h[i]):  # Not in strong 12h downtrend
                signals[i] = 0.25
                position = 1
            # Enter short: price at R3 (strong resistance) with volume, no strong 12h uptrend
            elif (close[i] >= R3_6h[i] * 0.995 and  # Near R3
                  close[i] <= R3_6h[i] * 1.005 and
                  volume_confirm and
                  not trend_up_12h_6h[i]):  # Not in strong 12h uptrend
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price reaches S4 (extreme) or volume drops
            if close[i] >= S4_6h[i] or not volume_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price reaches R4 (extreme) or volume drops
            if close[i] <= R4_6h[i] or not volume_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals