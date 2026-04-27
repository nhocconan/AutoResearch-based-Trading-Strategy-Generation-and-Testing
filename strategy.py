#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: Camarilla R1/S1 breakouts on 1h with 4h EMA21 trend filter and volume spike confirmation. Designed for 1h timeframe to achieve 60-150 total trades over 4 years (15-37/year). Uses discrete position sizing (0.20) to minimize fee drag. Works in both bull and bear markets by following 4h trend direction while using Camarilla R1/S1 levels for breakout confirmation. Volume spike filter reduces false breakouts. ATR-based trailing stop manages risk without look-ahead.
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
    
    # Get 4h data for EMA trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA21 for trend filter
    close_4h_series = pd.Series(df_4h['close'].values)
    ema_21_4h = close_4h_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Calculate 4h OHLC for Camarilla levels
    o_4h = df_4h['open'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Camarilla levels: R1/S1 from 4h OHLC (tighter breakout levels)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = c_4h + (h_4h - l_4h) * 1.1 / 12
    camarilla_s1 = c_4h - (h_4h - l_4h) * 1.1 / 12
    
    # Align 4h indicators to 1h timeframe (completed bars only)
    ema_21_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital (discrete level)
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: need 4h EMA21 (21) + volume avg (20)
    start_idx = max(21, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_21_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_21_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with 4h EMA21 trend filter AND volume spike
            # Long: price closes above R1 AND above EMA21 (4h uptrend) AND volume spike
            long_condition = (close_val > r1_val) and (close_val > ema_val) and vol_conf
            # Short: price closes below S1 AND below EMA21 (4h downtrend) AND volume spike
            short_condition = (close_val < s1_val) and (close_val < ema_val) and vol_conf
            
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
            # 1. Price touches S1 (opposite Camarilla level)
            # 2. 4h EMA21 turns bearish (price below EMA)
            # 3. ATR-based trailing stop: price drops 2.0 * ATR from highest since entry
            atr_val = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean().values[i]
            exit_condition = (close_val < s1_val) or (close_val < ema_val) or (close_val < highest_since_entry - 2.0 * atr_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, close_val)
            
            # Exit conditions:
            # 1. Price touches R1 (opposite Camarilla level)
            # 2. 4h EMA21 turns bullish (price above EMA)
            # 3. ATR-based trailing stop: price rises 2.0 * ATR from lowest since entry
            atr_val = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean().values[i]
            exit_condition = (close_val > r1_val) or (close_val > ema_val) or (close_val > lowest_since_entry + 2.0 * atr_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0