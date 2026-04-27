#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeConfirm
Hypothesis: On 6h timeframe, trade Donchian(20) breakouts only when aligned with weekly pivot direction (bullish/bearish) and confirmed by volume spike. Uses 1d timeframe for weekly pivot calculation (proxy for weekly via prior 1d OHLC aggregation) and 12h for trend filter. Designed for low trade frequency (12-37/year) with discrete sizing (0.25) to minimize fee drag. Weekly pivot direction provides structural bias that works in both bull and bear markets by identifying key institutional levels. Volume confirmation reduces false breakouts. ATR-based stop manages risk.
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
    
    # Get 1d data for weekly pivot calculation (using prior 1d OHLC as proxy for weekly structure)
    df_1d = get_htf_data(prices, '1d')
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate weekly pivot points from prior 1d OHLC (simplified: using prior 1d as weekly proxy)
    # Standard weekly pivot: P = (PriorWeek High + PriorWeek Low + PriorWeek Close) / 3
    # We'll use prior 1d OHLC as available proxy (not perfect but usable for structure)
    # In practice, we'd need actual weekly data, but 1d gives us the OHLC needed
    # For true weekly, we'd use get_htf_data(prices, '1w') but using 1d as proxy for now
    # Actually, let's use 1d to calculate a pseudo-weekly pivot from the prior 1d bar
    # This gives us a structure level that resets daily but still provides S/R
    # Better: use actual weekly data if available
    df_1w = get_htf_data(prices, '1w')  # Get actual weekly data
    
    # Weekly pivot from prior weekly OHLC
    # P = (PriorWeek High + PriorWeek Low + PriorWeek Close) / 3
    # R1 = 2*P - PriorWeek Low
    # S1 = 2*P - PriorWeek High
    # R2 = P + (PriorWeek High - PriorWeek Low)
    # S2 = P - (PriorWeek High - PriorWeek Low)
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    # Weekly pivot levels
    pivot_1w = (h_1w + l_1w + c_1w) / 3.0
    r1_1w = 2 * pivot_1w - l_1w
    s1_1w = 2 * pivot_1w - h_1w
    r2_1w = pivot_1w + (h_1w - l_1w)
    s2_1w = pivot_1w - (h_1w - l_1w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian(20) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # ATR(14) for trailing stop
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: need weekly pivot (1w), 12h EMA50 (50), Donchian (20), volume avg (20), ATR (14)
    start_idx = max(50, 20, 14)  # 50 is the highest
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        ema_val = ema_50_12h_aligned[i]
        d_high = donchian_high[i]
        d_low = donchian_low[i]
        vol_conf = volume_confirm[i]
        atr_val = atr[i]
        
        if position == 0:
            # Determine weekly pivot bias: bullish if price above pivot, bearish if below
            bullish_bias = close_val > pivot_val
            bearish_bias = close_val < pivot_val
            
            # Look for entry: Donchian breakout with weekly pivot bias AND 12h EMA filter AND volume spike
            # Long: price breaks above Donchian HIGH AND bullish bias AND above EMA50_12h AND volume spike
            long_condition = (close_val > d_high) and bullish_bias and (close_val > ema_val) and vol_conf
            # Short: price breaks below Donchian LOW AND bearish bias AND below EMA50_12h AND volume spike
            short_condition = (close_val < d_low) and bearish_bias and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
        elif position == 1:
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, close_val)
            
            # Exit conditions:
            # 1. Price touches Donchian LOW (opposite breakout level)
            # 2. Weekly pivot bias turns bearish (price below weekly pivot)
            # 3. ATR-based trailing stop: price drops 2.5 * ATR from highest since entry
            exit_condition = (close_val < d_low) or (close_val < pivot_val) or (close_val < highest_since_entry - 2.5 * atr_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, close_val)
            
            # Exit conditions:
            # 1. Price touches Donchian HIGH (opposite breakout level)
            # 2. Weekly pivot bias turns bullish (price above weekly pivot)
            # 3. ATR-based trailing stop: price rises 2.5 * ATR from lowest since entry
            exit_condition = (close_val > d_high) or (close_val > pivot_val) or (close_val > lowest_since_entry + 2.5 * atr_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeConfirm"
timeframe = "6h"
leverage = 1.0