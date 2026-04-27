#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_RegimeFilter_Volume
Hypothesis: Uses 4h timeframe with Camarilla R1/S1 breakouts filtered by 1d trend (price vs EMA50), choppiness regime filter (CHOP > 61.8 = range, < 38.2 = trend), and volume confirmation. Only takes breakouts in the direction of the 1d trend when market is trending (CHOP < 38.2). Designed to work in both bull and bear markets by avoiding range-bound conditions and using 1d EMA50 for trend direction. Target ~20-30 trades/year to minimize fee drag.
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
    
    # Get 1d data for EMA50 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Get 1d data for Camarilla levels (from previous completed 1d bar)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.1 / 12)
    s1 = prev_close - (rng * 1.1 / 12)
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d Chopiness Index (CHOP) regime filter
    # CHOP = 100 * log10(sum(ATR1, 14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = np.abs(df_1d['high'] - df_1d['low'])
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    # Avoid division by zero
    chop = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14) if np.any(range_14 > 0) else np.full_like(close, 50.0)
    chop = np.where(range_14 > 0, 100 * np.log10(sum_atr_14 / range_14) / np.log10(14), 50.0)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.0
    
    # Warmup: need 1d EMA50 (50), 1d shift(1) for Camarilla, vol avg (20), ATR14 (14), CHOP (14)
    start_idx = max(50 + 1, 1 + 1, 20, 14, 14)  # ~51 bars
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_50_aligned[i]
        chop_val = chop_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with 1d EMA50 alignment, volume confirmation, and trending regime (CHOP < 38.2)
            long_condition = (close_val > r1_val and 
                            close_val > ema_val and 
                            vol_conf and 
                            chop_val < 38.2)
            short_condition = (close_val < s1_val and 
                             close_val < ema_val and 
                             vol_conf and 
                             chop_val < 38.2)
            
            if long_condition:
                signals[i] = 0.30
                position = 1
                size = 0.30
            elif short_condition:
                signals[i] = -0.30
                position = -1
                size = 0.30
        elif position == 1:
            # Exit long: price crosses below 1d EMA50 (trend reversal) OR chop regime shifts to range (CHOP > 61.8)
            if close_val < ema_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
                size = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 1d EMA50 (trend reversal) OR chop regime shifts to range (CHOP > 61.8)
            if close_val > ema_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
                size = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_RegimeFilter_Volume"
timeframe = "4h"
leverage = 1.0