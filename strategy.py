#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter_v5
Hypothesis: 4h Camarilla R1/S1 breakouts with 1d EMA50 trend filter, volume spike confirmation, and choppiness regime filter. Designed for 4h timeframe to achieve 75-200 total trades over 4 years (19-50/year). Uses discrete position sizing (0.30) to balance profit potential and fee drag. Works in both bull and bear markets by following 1d trend direction while using Camarilla R1/S1 levels for breakout confirmation. Volume spike and chop regime filters reduce false breakouts. ATR-based trailing stop manages risk without look-ahead.
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
    
    # Get 1d data for EMA trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d OHLC for Camarilla levels
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla levels: R1/S1 from 1d OHLC (tighter levels for more precise entries)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = c_1d + (h_1d - l_1d) * 1.1 / 12
    camarilla_s1 = c_1d - (h_1d - l_1d) * 1.1 / 12
    
    # Align 1d indicators to 4h timeframe (completed bars only)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Choppiness regime filter: CHOP(14) < 61.8 = trending (favor breakouts), > 61.8 = range (avoid)
    # True Range = max(high-low, |high-previous close|, |low-previous close|)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low to avoid roll artifact
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of True Range over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Choppiness Index = 100 * log10(tr_sum_14 / (atr_14 * 14)) / log10(14)
    chop = 100 * np.log10(tr_sum_14 / (atr_14 * 14)) / np.log10(14)
    # Avoid division by zero or invalid values
    chop = np.where((atr_14 > 0) & (tr_sum_14 > 0), chop, 50.0)  # neutral when undefined
    # Trending market: CHOP < 61.8 (favor breakouts)
    chop_filter = chop < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.30   # Position size: 30% of capital (discrete level)
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: need 1d EMA50 (50) + volume avg (20) + ATR (14) + CHOP (14)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_confirm[i]) or 
            np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_50_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_conf = volume_confirm[i]
        chop_ok = chop_filter[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with 1d EMA50 trend filter AND volume spike AND chop filter
            # Long: price closes above R1 AND above EMA50 (1d uptrend) AND volume spike AND trending market
            long_condition = (close_val > r1_val) and (close_val > ema_val) and vol_conf and chop_ok
            # Short: price closes below S1 AND below EMA50 (1d downtrend) AND volume spike AND trending market
            short_condition = (close_val < s1_val) and (close_val < ema_val) and vol_conf and chop_ok
            
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
            # 2. 1d EMA50 turns bearish (price below EMA)
            # 3. ATR-based trailing stop: price drops 2.0 * ATR from highest since entry
            atr_val = atr_14[i]
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
            # 2. 1d EMA50 turns bullish (price above EMA)
            # 3. ATR-based trailing stop: price rises 2.0 * ATR from lowest since entry
            atr_val = atr_14[i]
            exit_condition = (close_val > r1_val) or (close_val > ema_val) or (close_val > lowest_since_entry + 2.0 * atr_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter_v5"
timeframe = "4h"
leverage = 1.0