#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_ChopFilter_VolumeSpike_New
Hypothesis: Camarilla R1/S1 breakouts on 4h with 1d trend filter (price vs EMA50), 
Choppiness Index regime filter (CHOP > 50 = range, CHOP < 50 = trend), and volume spike confirmation. 
In trending regimes (CHOP < 50): follow 1d EMA50 direction for breakouts. 
In ranging regimes (CHOP >= 50): fade Camarilla extremes (sell at R1, buy at S1). 
ATR trailing stop (2.0x) manages risk. Designed for 4h timeframe to achieve 75-200 total trades over 4 years.
Works in both bull and bear markets by adapting to regime: trend follow in trends, mean revert in ranges.
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
    
    # Get 1d data for trend filter and Camarilla levels
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
    
    # Camarilla levels: R1/S1 from 1d OHLC
    camarilla_r1 = c_1d + (h_1d - l_1d) * 1.1 / 12
    camarilla_s1 = c_1d - (h_1d - l_1d) * 1.1 / 12
    
    # Align 1d indicators to 4h timeframe (completed bars only)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)  # already done above
    
    # Choppiness Index (14-period) on 1d to determine regime
    # CHOP = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    tr1 = h_1d - l_1d
    tr2 = np.abs(h_1d - np.roll(c_1d, 1))
    tr3 = np.abs(l_1d - np.roll(c_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (atr_1d * 14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # ATR for adaptive trailing stop (14-period ATR on 4h)
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr1_4h[0] = 0
    tr2_4h[0] = 0
    tr3_4h[0] = 0
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: need 1d EMA50 (50) + CHOP (14) + volume avg (20) + ATR (14)
    start_idx = max(50, 14, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_50_1d_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        chop_val = chop_aligned[i]
        vol_conf = volume_confirm[i]
        atr_val = atr_4h[i]
        
        if position == 0:
            # Regime-dependent entry logic
            if chop_val < 50:  # Trending regime: follow 1d EMA50 direction
                # Long: price closes above R1 AND above EMA50 (1d uptrend) AND volume spike
                long_condition = (close_val > r1_val) and (close_val > ema_val) and vol_conf
                # Short: price closes below S1 AND below EMA50 (1d downtrend) AND volume spike
                short_condition = (close_val < s1_val) and (close_val < ema_val) and vol_conf
            else:  # Ranging regime (CHOP >= 50): fade extremes
                # Long: price closes below S1 (oversold) AND volume spike
                long_condition = (close_val < s1_val) and vol_conf
                # Short: price closes above R1 (overbought) AND volume spike
                short_condition = (close_val > r1_val) and vol_conf
            
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
            # 1. Price touches opposite Camarilla level (S1 for long)
            # 2. 1d EMA50 turns against position (price below EMA for long)
            # 3. ATR-based trailing stop: price drops 2.0 * ATR from highest since entry
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
            # 1. Price touches opposite Camarilla level (R1 for short)
            # 2. 1d EMA50 turns against position (price above EMA for short)
            # 3. ATR-based trailing stop: price rises 2.0 * ATR from lowest since entry
            exit_condition = (close_val > r1_val) or (close_val > ema_val) or (close_val > lowest_since_entry + 2.0 * atr_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_ChopFilter_VolumeSpike_New"
timeframe = "4h"
leverage = 1.0