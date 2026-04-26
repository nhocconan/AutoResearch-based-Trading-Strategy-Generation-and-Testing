#!/usr/bin/env python3
"""
6h_ADX_Regime_Adaptive_Camarilla_R3_S3
Hypothesis: Use 1d ADX to detect regime (ADX>25 = trend, ADX<20 = range). In trend regime: breakout of Camarilla R3/S3 with volume spike. In range regime: mean reversion at Camarilla S3/R3 (buy at S3, sell at R3). Uses 6h timeframe for entries with 1d HTF for regime and levels. Designed for 50-150 trades over 4 years (12-37/year) to minimize fee drag. Works in bull/bear markets by adapting strategy to regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for regime and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ADX for regime detection (ADX > 25 = trend, ADX < 20 = range)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Smoothed TR, DM+,
    tr_period = 14
    tr_smooth = pd.Series(tr).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Previous day's OHLC for Camarilla levels
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d_prev = df_1d['high'].values
    low_1d_prev = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day's OHLC
    rng = high_1d_prev - low_1d_prev
    camarilla_r3 = close_1d_prev + (rng * 1.1 / 4)
    camarilla_s3 = close_1d_prev - (rng * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (20 for vol, 30 for ADX)
    start_idx = max(20, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            bars_since_entry += 1 if position != 0 else 0
            continue
        
        close_val = close[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        
        # Regime detection
        is_trend = adx_val > 25
        is_range = adx_val < 20
        
        # Entry conditions based on regime
        if is_trend:
            # Trend regime: breakout of Camarilla R3/S3 with volume spike
            long_entry = (close_val > camarilla_r3_val) and vol_spike
            short_entry = (close_val < camarilla_s3_val) and vol_spike
        elif is_range:
            # Range regime: mean reversion at Camarilla levels
            long_entry = (close_val <= camarilla_s3_val) and (close[i-1] > camarilla_s3_val if i>0 else False) and vol_spike
            short_entry = (close_val >= camarilla_r3_val) and (close[i-1] < camarilla_r3_val if i>0 else False) and vol_spike
        else:
            # Transition regime (ADX between 20-25): no entries
            long_entry = False
            short_entry = False
        
        # Exit conditions
        if position == 1:
            # Long exit: touch opposite level (S3) or regime change to range with reversal signal
            exit_long = (close_val < camarilla_s3_val) or (is_range and close_val > camarilla_r3_val * 0.999)
        elif position == -1:
            # Short exit: touch opposite level (R3) or regime change to range with reversal signal
            exit_short = (close_val > camarilla_r3_val) or (is_range and close_val < camarilla_s3_val * 1.001)
        else:
            exit_long = False
            exit_short = False
        
        # Minimum holding period: 3 bars
        min_hold = 3
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -base_size
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = base_size
                bars_since_entry += 1
        elif position == -1:
            # Short - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -base_size
                bars_since_entry += 1
    
    return signals

name = "6h_ADX_Regime_Adaptive_Camarilla_R3_S3"
timeframe = "6h"
leverage = 1.0