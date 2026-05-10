#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_1wVolumeConfirmation
# Hypothesis: Combining 1d trend (EMA34) and 1w volume confirmation with Camarilla breakouts on 12h
# provides high-probability entries aligned with higher timeframe momentum and institutional activity.
# Target: 12-30 trades/year to minimize fee drag on 12h timeframe, working in both bull and bear markets.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_1wVolumeConfirmation"
timeframe = "12h"
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
    
    # 1d trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align 1d trend to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # 1w volume confirmation (above 1.5x 4-week average)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    vol_ma_1w = np.zeros_like(volume_1w)
    vol_sum = 0
    for i in range(len(volume_1w)):
        vol_sum += volume_1w[i]
        if i >= 4:
            vol_sum -= volume_1w[i-4]
        if i >= 3:
            vol_ma_1w[i] = vol_sum / 4
        else:
            vol_ma_1w[i] = np.nan
    volume_confirm_1w = volume_1w > (1.5 * vol_ma_1w)
    volume_confirm_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_confirm_1w.astype(float))
    
    # Calculate Camarilla levels from previous 12h bar
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Shift to get previous 12h bar values
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels (R1, S1, R2, S2)
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    R2 = prev_close + (prev_high - prev_low) * 1.1 / 6
    S2 = prev_close - (prev_high - prev_low) * 1.1 / 6
    
    # Align Camarilla levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_12h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_12h, S1)
    R2_aligned = align_htf_to_ltf(prices, df_12h, R2)
    S2_aligned = align_htf_to_ltf(prices, df_12h, S2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(volume_confirm_1w_aligned[i]) or
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with 1d uptrend and 1w volume confirmation
            if (high[i] > R1_aligned[i] and
                trend_1d_up_aligned[i] > 0.5 and
                volume_confirm_1w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with 1d downtrend and 1w volume confirmation
            elif (low[i] < S1_aligned[i] and
                  trend_1d_down_aligned[i] > 0.5 and
                  volume_confirm_1w_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below S1 or 1d trend turns down
            if (low[i] < S1_aligned[i] or
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above R1 or 1d trend turns up
            if (high[i] > R1_aligned[i] or
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals