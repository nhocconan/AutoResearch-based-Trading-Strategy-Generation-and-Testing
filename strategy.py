#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with daily Camarilla pivot level touch + volume confirmation + ADX trend filter
# Strategy: Long when price touches Camarilla L3 support with volume > 1.5x average and ADX > 25
# Short when price touches Camarilla H3 resistance with volume > 1.5x average and ADX > 25
# Uses daily pivot levels for mean reversion in ranging markets, ADX to avoid strong trends
# Volume surge confirms reversal strength at support/resistance
# Target: 20-50 total trades over 4 years (5-12.5/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels
    # Pivot = (High + Low + Close) / 3
    # Range = High - Low
    # H3 = Pivot + 1.1 * Range / 2
    # L3 = Pivot - 1.1 * Range / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    h3_1d = pivot_1d + 1.1 * range_1d / 2.0
    l3_1d = pivot_1d - 1.1 * range_1d / 2.0
    
    # Calculate ADX (14) on daily for trend strength
    # +DM, -DM, TR
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    tr1 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr2 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr3 = np.abs(high_1d[1:] - low_1d[:-1])
    tr3 = np.concatenate([[tr3[0]] if len(tr3) > 0 else [0], tr3])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 4h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(h3_1d_aligned[i]) or 
            np.isnan(l3_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume surge condition
        volume_surge = volume[i] > 1.5 * vol_ma_20[i]
        
        # Camarilla level touch conditions (with small tolerance)
        tolerance = 0.001  # 0.1% tolerance for level touch
        long_touch = abs(close[i] - l3_1d_aligned[i]) / l3_1d_aligned[i] < tolerance
        short_touch = abs(close[i] - h3_1d_aligned[i]) / h3_1d_aligned[i] < tolerance
        
        # Trend filter: avoid strong trends (ADX < 25 for mean reversion)
        weak_trend = adx_1d_aligned[i] < 25
        
        # Entry logic
        long_entry = long_touch and volume_surge and weak_trend
        short_entry = short_touch and volume_surge and weak_trend
        
        # Exit conditions: opposite touch or trend strengthens
        exit_long = position == 1 and (short_touch or adx_1d_aligned[i] > 30)
        exit_short = position == -1 and (long_touch or adx_1d_aligned[i] > 30)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_touch_volume_adx_v1"
timeframe = "4h"
leverage = 1.0