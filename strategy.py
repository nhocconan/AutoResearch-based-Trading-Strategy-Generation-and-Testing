#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14115_6d_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot points (once before loop)
    df_w = get_htf_data(prices, '1w')
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points (using previous week)
    # Pivot = (H + L + C) / 3
    # R1 = (2 * P) - L
    # S1 = (2 * P) - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    # R3 = H + 2*(P - L)
    # S3 = L - 2*(H - P)
    H = high_w
    L = low_w
    C = close_w
    pivot = (H + L + C) / 3
    r1 = (2 * pivot) - L
    s1 = (2 * pivot) - H
    r2 = pivot + (H - L)
    s2 = pivot - (H - L)
    r3 = H + 2 * (pivot - L)
    s3 = L - 2 * (H - pivot)
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_w, s3)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for volume, 14 for ATR)
    start = max(20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or \
           np.isnan(s3_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Weekly pivot-based signals with volume confirmation
        # Bounce at S1/S2: Long
        # Breakdown below S2: Short
        # Bounce at R1/R2: Short
        # Breakout above R2: Long
        bounce_s1 = (close[i] > s1_aligned[i-1]) and (close[i-1] <= s1_aligned[i-1]) and vol_filter[i]
        bounce_s2 = (close[i] > s2_aligned[i-1]) and (close[i-1] <= s2_aligned[i-1]) and vol_filter[i]
        breakdown_s2 = (close[i] < s2_aligned[i-1]) and vol_filter[i]
        bounce_r1 = (close[i] < r1_aligned[i-1]) and (close[i-1] >= r1_aligned[i-1]) and vol_filter[i]
        bounce_r2 = (close[i] < r2_aligned[i-1]) and (close[i-1] >= r2_aligned[i-1]) and vol_filter[i]
        breakout_r2 = (close[i] > r2_aligned[i-1]) and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if bounce_s1 or bounce_s2:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif breakdown_s2:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            elif bounce_r1 or bounce_r2:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            elif breakout_r2:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or reversal at R1/R2
            if close[i] <= stop_price or bounce_r1 or bounce_r2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or reversal at S1/S2
            if close[i] >= stop_price or bounce_s1 or bounce_s2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals