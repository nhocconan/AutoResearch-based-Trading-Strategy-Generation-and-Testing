#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Regime_v1
Hypothesis: Camarilla R3/S3 breakouts with volume confirmation and 1d EMA34 trend filter, plus Choppiness regime filter (CHOP > 61.8 = range, only trade reversals at S3/R3; CHOP < 38.2 = trend, trade breakouts). Designed to work in both bull and bear markets by adapting to regime.
"""

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Regime_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation (using previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend filter
    df_1d_ema = get_htf_data(prices, '1d')
    if len(df_1d_ema) < 34:
        return np.zeros(n)
    
    # Get 1d data for Choppiness calculation
    df_1d_chop = get_htf_data(prices, '1d')
    if len(df_1d_chop) < 14:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R3 and S3 levels
    rng = prev_high - prev_low
    r3 = prev_close + (rng * 1.1 / 4)   # R3 = C + (H-L) * 1.1/4
    s3 = prev_close - (rng * 1.1 / 4)   # S3 = C - (H-L) * 1.1/4
    
    # Align 1d levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d_ema['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d_ema, ema_34_1d)
    
    # Calculate Choppiness Index (14-period) on 1d
    high_1d = df_1d_chop['high'].values
    low_1d = df_1d_chop['low'].values
    close_1d = df_1d_chop['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close_1d
    
    # ATR14
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Max high and min low over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (max_high - min_low)) / log10(14)
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    denominator = max_high - min_low
    chop = 100 * np.log10(sum_tr / denominator) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d_chop, chop)
    
    # Volume confirmation (20-period MA on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34), Chop (14), volume MA (20)
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend filter
        uptrend_1d = close[i] > ema_34_1d_aligned[i]
        downtrend_1d = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation (>2x average volume)
        volume_confirm = volume[i] > volume_ma[i] * 2.0
        
        # Regime filter
        chop_val = chop_aligned[i]
        ranging_market = chop_val > 61.8  # CHOP > 61.8 = range
        trending_market = chop_val < 38.2  # CHOP < 38.2 = trend
        
        if position == 0:
            # Long entry conditions
            long_breakout = close[i] > r3_aligned[i]
            long_reversal = close[i] > s3_aligned[i] and ranging_market  # Buy at S3 in range
            
            # Short entry conditions
            short_breakout = close[i] < s3_aligned[i]
            short_reversal = close[i] < r3_aligned[i] and ranging_market  # Sell at R3 in range
            
            # In trending market: trade breakouts
            # In ranging market: trade reversals at S3/R3
            if (trending_market and long_breakout and uptrend_1d and volume_confirm) or \
               (ranging_market and long_reversal and volume_confirm):
                signals[i] = 0.25
                position = 1
            elif (trending_market and short_breakout and downtrend_1d and volume_confirm) or \
                 (ranging_market and short_reversal and volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or reversal signal
            exit_condition = (not uptrend_1d) or \
                            (ranging_market and close[i] < r3_aligned[i]) or \
                            (trending_market and close[i] < s3_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or reversal signal
            exit_condition = (not downtrend_1d) or \
                            (ranging_market and close[i] > s3_aligned[i]) or \
                            (trending_market and close[i] > r3_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals