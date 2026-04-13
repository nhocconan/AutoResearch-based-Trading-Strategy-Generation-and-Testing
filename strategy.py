#!/usr/bin/env python3
"""
Hypothesis: 4h 1-day/1-week Camarilla pivot reversal with volume confirmation and ADX trend filter.
Uses 1-day Camarilla pivot levels (H4, L4) for reversal entries in ranging markets, confirmed by
1-day volume spikes and 1-week ADX < 25 (low trend strength) to avoid false signals in strong trends.
Targets 30-80 total trades over 4 years (7-20/year) to minimize fee drag while capturing mean reversion.
"""

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
    
    # Get 1d data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # H4 = C + 1.1/2 * Range
    # L4 = C - 1.1/2 * Range
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + 1.1 / 2.0 * range_1d
    camarilla_l4 = close_1d - 1.1 / 2.0 * range_1d
    
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 1-day volume spike (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1-week ADX (14-period)
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])], 
                        np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # ADX < 25 = low trend strength (good for mean reversion)
    low_trend = adx < 25
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    low_trend_aligned = align_htf_to_ltf(prices, df_1w, low_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(low_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Camarilla H4/L4 touch + volume spike + low trend
        touch_h4 = close[i] >= camarilla_h4_aligned[i] * 0.999  # Allow small slippage
        touch_l4 = close[i] <= camarilla_l4_aligned[i] * 1.001  # Allow small slippage
        vol_confirm = vol_spike_aligned[i] > 0.5  # True if volume spike
        trend_filter = low_trend_aligned[i] > 0.5  # True if low trend (ADX < 25)
        
        long_entry = touch_l4 and vol_confirm and trend_filter
        short_entry = touch_h4 and vol_confirm and trend_filter
        
        # Exit when price reaches opposite Camarilla level (H3/L3)
        camarilla_h3 = close_1d + 1.1/4.0 * range_1d
        camarilla_l3 = close_1d - 1.1/4.0 * range_1d
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        
        exit_long = position == 1 and close[i] >= camarilla_h3_aligned[i]
        exit_short = position == -1 and close[i] <= camarilla_l3_aligned[i]
        
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

name = "4h_1d_1w_camarilla_reversal"
timeframe = "4h"
leverage = 1.0