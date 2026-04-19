# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 4h-based breakout strategy using 12h pivot points (R1/S1) with volume confirmation and ADX trend filter.
- Uses 12h pivot levels as dynamic support/resistance (more stable than daily/weekly)
- Enters long on break above R1 with volume > 1.5x 20-period average and ADX > 25 (trending)
- Enters short on break below S1 with same conditions
- Exits when price returns to 12h pivot level
- Designed to work in both bull (breakouts) and bear (breakdowns) markets
- Target: 20-40 trades/year to avoid fee drag while capturing strong moves
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Pivot_R1S1_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for pivot points (once before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h pivot points (standard formula)
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    r1_12h = 2 * pivot_12h - low_12h
    s1_12h = 2 * pivot_12h - high_12h
    
    # Align 12h pivot levels to 4h timeframe
    pivot_4h = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_4h = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_4h = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX trend filter (14-period)
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        result[period-1] = np.nanmean(data[:period])  # Simple average for first value
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di14 = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    dx = np.where((plus_di14 + minus_di14) != 0, 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilders_smoothing(dx, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Enough warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(pivot_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or \
           np.isnan(vol_ma_20[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        trend_strength = adx[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        trending = trend_strength > 25  # ADX > 25 indicates trending market
        
        if position == 0:
            # Long: Price breaks above R1 with volume and trend
            if price > r1_4h[i] and volume_confirmed and trending:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume and trend
            elif price < s1_4h[i] and volume_confirmed and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns to pivot level (mean reversion within trend)
            if price < pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns to pivot level
            if price > pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals