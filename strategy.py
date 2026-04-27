#!/usr/bin/env python3
"""
1d_Donchian20_WeeklyTrend_VolumeConfirm_ATRStop
Hypothesis: Daily Donchian(20) breakouts with weekly EMA50 trend filter and volume confirmation. ATR-based trailing stop manages risk. Designed for 1d timeframe to achieve 30-100 total trades over 4 years (7-25/year). Works in bull markets via breakouts above upper band and in bear markets via breakouts below lower band, filtered by weekly trend direction. Volume confirmation reduces false breakouts. Discrete position sizing (0.25) minimizes fee drag.
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
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: need weekly EMA50 (50) + Donchian (20) + volume avg (20)
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_50_1w_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with weekly EMA50 trend filter AND volume confirmation
            # Long: price closes above upper channel AND above weekly EMA50 (uptrend) AND volume confirmation
            long_condition = (close_val > upper_channel) and (close_val > ema_val) and vol_conf
            # Short: price closes below lower channel AND below weekly EMA50 (downtrend) AND volume confirmation
            short_condition = (close_val < lower_channel) and (close_val < ema_val) and vol_conf
            
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
            # 1. Price touches lower Donchian channel (opposite breakout)
            # 2. Weekly EMA50 turns bearish (price below EMA)
            # 3. ATR-based trailing stop: price drops 2.5 * ATR from highest since entry
            atr_val = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean().values[i]
            exit_condition = (close_val < lower_channel) or (close_val < ema_val) or (close_val < highest_since_entry - 2.5 * atr_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, close_val)
            
            # Exit conditions:
            # 1. Price touches upper Donchian channel (opposite breakout)
            # 2. Weekly EMA50 turns bullish (price above EMA)
            # 3. ATR-based trailing stop: price rises 2.5 * ATR from lowest since entry
            atr_val = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean().values[i]
            exit_condition = (close_val > upper_channel) or (close_val > ema_val) or (close_val > lowest_since_entry + 2.5 * atr_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_WeeklyTrend_VolumeConfirm_ATRStop"
timeframe = "1d"
leverage = 1.0