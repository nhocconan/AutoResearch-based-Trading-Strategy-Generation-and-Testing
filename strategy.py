#!/usr/bin/env python3
"""
6h Camarilla R3S3 Breakout + Weekly Pivot Direction + Volume Confirmation
Hypothesis: Weekly pivot levels (from prior week) establish major support/resistance. 
6h Camarilla R3/S3 breakouts aligned with weekly pivot direction capture institutional flow. 
Volume confirmation filters false breakouts. Works in bull/bear via discrete sizing (0.25) and 
weekly trend filter. Primary 6h timeframe targets 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly pivot levels from prior completed week (H, L, C of week-1)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    weekly_range = prev_week_high - prev_week_low
    weekly_R1 = 2 * weekly_pivot - prev_week_low
    weekly_S1 = 2 * weekly_pivot - prev_week_high
    weekly_R2 = weekly_pivot + weekly_range
    weekly_S2 = weekly_pivot - weekly_range
    weekly_R3 = prev_week_high + 2 * (weekly_pivot - prev_week_low)
    weekly_S3 = prev_week_low - 2 * (prev_week_high - weekly_pivot)
    
    # Align weekly pivots to 6h (they update only when new weekly bar completes)
    weekly_R3_aligned = align_htf_to_ltf(prices, df_1w, weekly_R3)
    weekly_S3_aligned = align_htf_to_ltf(prices, df_1w, weekly_S3)
    weekly_trend_up = weekly_close > weekly_open  # need weekly open
    weekly_open = df_1w['open'].shift(1).values
    weekly_trend_up = prev_week_close > weekly_open
    weekly_trend_down = prev_week_close < weekly_open
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    
    # 1d Camarilla levels from previous 1d bar
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    day_range = prev_day_high - prev_day_low
    camarilla_R3 = prev_day_close + day_range * 1.1 / 4
    camarilla_S3 = prev_day_close - day_range * 1.1 / 4
    
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for weekly and daily warmup
    start_idx = max(30, 20)  # weekly needs 1 bar shift + buffer, daily similar
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_R3_aligned[i]) or np.isnan(weekly_S3_aligned[i]) or
            np.isnan(weekly_trend_up_aligned[i]) or np.isnan(weekly_trend_down_aligned[i]) or
            np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above camarilla_R3 AND above weekly_S3 (bullish weekly context) AND volume spike
            long_entry = (curr_high > camarilla_R3_aligned[i]) and \
                         (curr_close > weekly_S3_aligned[i]) and \
                         weekly_trend_up_aligned[i] and \
                         vol_spike
            # Short: break below camarilla_S3 AND below weekly_R3 (bearish weekly context) AND volume spike
            short_entry = (curr_low < camarilla_S3_aligned[i]) and \
                          (curr_close < weekly_R3_aligned[i]) and \
                          weekly_trend_down_aligned[i] and \
                          vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: break below camarilla_S3 OR weekly trend turns down
            if (curr_low < camarilla_S3_aligned[i]) or (~weekly_trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above camarilla_R3 OR weekly trend turns up
            if (curr_high > camarilla_R3_aligned[i]) or (weekly_trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_WeeklyPivotDir_VolumeConfirm"
timeframe = "6h"
leverage = 1.0