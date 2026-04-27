#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_RegimeFilter
Hypothesis: Camarilla R1/S1 breakouts on 12h with 1w EMA50 trend filter, volume spike confirmation, and choppiness regime filter. Uses discrete position sizing (0.25) to reduce fee drag. Designed for 12h timeframe to achieve 50-150 total trades over 4 years (12-37/year). Works in both bull and bear markets by following 1w trend direction while using Camarilla levels for precise entries. Added choppiness regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trend) to avoid whipsaws in sideways markets. Volume spike filter reduces false breakouts. ATR-based trailing stop manages risk without look-ahead.
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
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d OHLC for Camarilla levels
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla levels: R1/S1 from 1d OHLC (tighter than R3/S3 for better precision)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = c_1d + (h_1d - l_1d) * 1.1 / 12
    camarilla_s1 = c_1d - (h_1d - l_1d) * 1.1 / 12
    
    # Align 1d indicators to 12h timeframe (completed bars only)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (stricter to reduce trades)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Choppiness Index regime filter (14-period)
    # CHOP > 61.8 = ranging market (mean revert), CHOP < 38.2 = trending market (trend follow)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop[np.isnan(chop) | (highest_high == lowest_low)] = 50  # neutral when range is zero
    
    # Regime: True when trending (CHOP < 38.2), False when ranging (CHOP > 61.8)
    trending_regime = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: need 1w EMA50 (50) + volume avg (20) + ATR for CHOP (14)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_confirm[i]) or np.isnan(trending_regime[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_50_1w_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_conf = volume_confirm[i]
        regime = trending_regime[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with 1w EMA50 trend filter, volume spike, AND trending regime
            # Long: price closes above R1 AND above EMA50 (1w uptrend) AND volume spike AND trending regime
            long_condition = (close_val > r1_val) and (close_val > ema_val) and vol_conf and regime
            # Short: price closes below S1 AND below EMA50 (1w downtrend) AND volume spike AND trending regime
            short_condition = (close_val < s1_val) and (close_val < ema_val) and vol_conf and regime
            
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
            # 2. 1w EMA50 turns bearish (price below EMA)
            # 3. ATR-based trailing stop: price drops 2.5 * ATR from highest since entry
            atr_val = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean().values[i]
            exit_condition = (close_val < s1_val) or (close_val < ema_val) or (close_val < highest_since_entry - 2.5 * atr_val)
            
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
            # 2. 1w EMA50 turns bullish (price above EMA)
            # 3. ATR-based trailing stop: price rises 2.5 * ATR from lowest since entry
            atr_val = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean().values[i]
            exit_condition = (close_val > r1_val) or (close_val > ema_val) or (close_val > lowest_since_entry + 2.5 * atr_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_RegimeFilter"
timeframe = "12h"
leverage = 1.0