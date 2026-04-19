#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Pivot Reversal Strategy with Weekly ADX filter and Volume Confirmation
# Uses weekly pivot points (R1, S1) as key support/resistance levels
# Weekly ADX > 25 filters for trending conditions to avoid whipsaws in ranging markets
# Volume spike (>1.5x 24-period average) confirms breakout validity
# Target: 15-25 trades/year per symbol with disciplined entries
# Works in bull markets (buy S1 bounces, sell R1 rejections) and bear markets (sell R1 breaks, buy S1 breaks)
name = "12h_PivotReversal_WeeklyADX_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot points and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = (2 * Pivot) - L
    # S1 = (2 * Pivot) - H
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = (2 * pivot) - weekly_low
    s1 = (2 * pivot) - weekly_high
    
    # Align weekly pivot levels to 12h timeframe (wait for weekly bar close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Weekly ADX for trend strength filtering
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # First value is NaN
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smoothed values
        def smooth_wilder(values, period):
            smoothed = np.full_like(values, np.nan)
            if len(values) >= period:
                smoothed[period-1] = np.nansum(values[:period])
                for i in range(period, len(values)):
                    smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
            return smoothed
        
        atr = smooth_wilder(tr, period)
        dm_plus_smooth = smooth_wilder(dm_plus, period)
        dm_minus_smooth = smooth_wilder(dm_minus, period)
        
        # DI+ and DI-
        di_plus = np.where(atr != 0, (dm_plus_smooth / atr) * 100, 0)
        di_minus = np.where(atr != 0, (dm_minus_smooth / atr) * 100, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) != 0, 
                      np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
        adx = smooth_wilder(dx, period)
        return adx
    
    adx = calculate_adx(weekly_high, weekly_low, weekly_close, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: volume > 1.5 * 24-period average (2 weeks of 12h data)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending markets (ADX > 25)
        if adx_aligned[i] <= 25:
            # In ranging markets, stay flat or use mean reversion at extremes
            if position == 1 and close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long signals: price breaks above S1 with volume OR bounces off S1
            if ((close[i] > s1_aligned[i] and close[i-1] <= s1_aligned[i-1] and volume_spike[i]) or
                (close[i] >= s1_aligned[i] * 0.995 and close[i] <= s1_aligned[i] * 1.005 and 
                 close[i-1] < s1_aligned[i-1] and volume_spike[i])):
                signals[i] = 0.25
                position = 1
            # Short signals: price breaks below R1 with volume OR rejection at R1
            elif ((close[i] < r1_aligned[i] and close[i-1] >= r1_aligned[i-1] and volume_spike[i]) or
                  (close[i] <= r1_aligned[i] * 1.005 and close[i] >= r1_aligned[i] * 0.995 and
                   close[i-1] > r1_aligned[i-1] and volume_spike[i])):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: price breaks below pivot or reaches R1
            if close[i] < pivot_aligned[i] or close[i] >= r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: price breaks above pivot or reaches S1
            if close[i] > pivot_aligned[i] or close[i] <= s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals