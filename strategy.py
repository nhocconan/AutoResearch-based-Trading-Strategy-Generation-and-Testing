#!/usr/bin/env python3
"""
6h Weekly Pivot Direction + Volume Spike + ADX Trend Filter
Based on weekly pivot levels from higher timeframe (1w) to establish bias.
Long when price breaks above weekly R1 with volume spike and ADX > 20.
Short when price breaks below weekly S1 with volume spike and ADX > 20.
Uses ADX(14) on 6h to filter ranging markets and ensure trending conditions.
Designed for low trade frequency with clear trend-following edge in both bull and bear markets.
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
    
    # Get weekly data for pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (standard formula)
    # Using previous week's OHLC for current week's pivot
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    
    # Pivot = (H + L + C) / 3
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # R1 = (2 * P) - L
    r1 = (2 * pivot) - weekly_low
    # S1 = (2 * P) - H
    s1 = (2 * pivot) - weekly_high
    
    # Align weekly levels to 6h timeframe (with proper delay for weekly bar close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Calculate ADX(14) on 6h for trend strength filter
    # +DM and -DM calculation
    high_diff = np.diff(high, prepend=high[0])
    low_diff = -np.diff(low, prepend=low[0])  # negative of low diff
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first bar
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            # Wilder smoothing: today = alpha * today + (1-alpha) * yesterday
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    atr = wilders_smoothing(tr, period)
    plus_di = 100 * wilders_smoothing(plus_dm, period) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, period) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # Volume spike detection (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need enough history for ADX and other calculations
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx[i]
        
        # Only trade when ADX > 20 (trending market)
        if adx_val > 20:
            if position == 0:
                # Long: price breaks above weekly R1 with volume spike
                if price > r1_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below weekly S1 with volume spike
                elif price < s1_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
            
            elif position == 1:
                # Maintain long position
                signals[i] = 0.25
                # Exit: price breaks below weekly pivot or ADX weakens
                if price < pivot_aligned[i] or adx_val < 15:
                    signals[i] = 0.0
                    position = 0
            
            elif position == -1:
                # Maintain short position
                signals[i] = -0.25
                # Exit: price breaks above weekly pivot or ADX weakens
                if price > pivot_aligned[i] or adx_val < 15:
                    signals[i] = 0.0
                    position = 0
        else:
            # In ranging market (ADX <= 20), stay flat
            signals[i] = 0.0
            position = 0
    
    return signals

name = "6h_WeeklyPivot_Direction_Volume_ADXFilter"
timeframe = "6h"
leverage = 1.0