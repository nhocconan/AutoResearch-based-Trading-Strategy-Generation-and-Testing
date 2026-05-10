#!/usr/bin/env python3
# 6h_Weekly_Pivot_DailyTrend_VolumeBreakout
# Hypothesis: 6s breakout of weekly Camarilla R4/S4 levels with 1d EMA34 trend filter and volume spike confirmation.
# Uses 1d trend for bias to avoid whipsaws in sideways markets, 6h for entry timing.
# Targets 15-25 trades/year to minimize fee drag. Works in bull/bear by trading breakouts aligned with higher timeframe trend.

name = "6h_Weekly_Pivot_DailyTrend_VolumeBreakout"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align 1d trend to 6h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # 1w data for Camarilla pivot levels (using previous week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Shift to get previous week's values (avoid look-ahead)
    prev_close = np.concatenate([[close_1w[0]], close_1w[:-1]])
    prev_high = np.concatenate([[high_1w[0]], high_1w[:-1]])
    prev_low = np.concatenate([[low_1w[0]], low_1w[:-1]])
    
    # Calculate weekly Camarilla levels
    # R4 = Close + (High - Low) * 1.1/2
    # S4 = Close - (High - Low) * 1.1/2
    camarilla_r4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align weekly Camarilla levels to 6h
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Volume filter: current volume > 2.0 * 24-period average (4 days of 6h bars)
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_filter = vol_ratio > 2.0
        
        if position == 0:
            # Long: price breaks above weekly Camarilla R4 with uptrend and volume spike
            if (close[i] > camarilla_r4_aligned[i] and
                trend_1d_up_aligned[i] > 0.5 and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Camarilla S4 with downtrend and volume spike
            elif (close[i] < camarilla_s4_aligned[i] and
                  trend_1d_down_aligned[i] > 0.5 and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to weekly Camarilla pivot (central level) or trend fails
            camarilla_pivot = (prev_high[i] + prev_low[i] + prev_close[i]) / 3
            camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, np.full_like(prev_close, camarilla_pivot))[i] if not np.isnan(prev_high[i]) else camarilla_pivot
            
            if (close[i] < camarilla_pivot_aligned or
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to weekly Camarilla pivot or trend fails
            camarilla_pivot = (prev_high[i] + prev_low[i] + prev_close[i]) / 3
            camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, np.full_like(prev_close, camarilla_pivot))[i] if not np.isnan(prev_high[i]) else camarilla_pivot
            
            if (close[i] > camarilla_pivot_aligned or
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals