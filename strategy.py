#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1
# Hypothesis: 12h breakout of weekly Camarilla R1/S1 levels with 1w trend filter and volume spike confirmation.
# Uses 1w trend for bias to avoid whipsaws in sideways markets, 12h for entry timing.
# Targets 15-30 trades/year to minimize fee drag. Works in bull/bear by trading breakouts aligned with higher timeframe trend.
# Added volume confirmation to reduce false breakouts.

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1"
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
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA34 trend
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w_up = close_1w > ema34_1w
    trend_1w_down = close_1w < ema34_1w
    
    # Align 1w trend to 12h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # 1w data for Camarilla pivot levels (using previous week's OHLC)
    close_1w_arr = df_1w['close'].values
    high_1w_arr = df_1w['high'].values
    low_1w_arr = df_1w['low'].values
    
    # Shift to get previous week's values (avoid look-ahead)
    prev_close = np.concatenate([[close_1w_arr[0]], close_1w_arr[:-1]])
    prev_high = np.concatenate([[high_1w_arr[0]], high_1w_arr[:-1]])
    prev_low = np.concatenate([[low_1w_arr[0]], low_1w_arr[:-1]])
    
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 12h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with uptrend and volume spike
            if (close[i] > camarilla_r1_aligned[i] and
                trend_1w_up_aligned[i] > 0.5 and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 with downtrend and volume spike
            elif (close[i] < camarilla_s1_aligned[i] and
                  trend_1w_down_aligned[i] > 0.5 and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to Camarilla pivot (central level) or trend fails
            camarilla_pivot = (prev_high[i] + prev_low[i] + prev_close[i]) / 3
            camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, np.full_like(prev_close, camarilla_pivot))[i] if not np.isnan(prev_high[i]) else camarilla_pivot
            
            if (close[i] < camarilla_pivot_aligned or
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to Camarilla pivot or trend fails
            camarilla_pivot = (prev_high[i] + prev_low[i] + prev_close[i]) / 3
            camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, np.full_like(prev_close, camarilla_pivot))[i] if not np.isnan(prev_high[i]) else camarilla_pivot
            
            if (close[i] > camarilla_pivot_aligned or
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals