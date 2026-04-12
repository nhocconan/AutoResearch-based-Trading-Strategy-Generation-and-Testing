#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_range_reversion_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least 5 days for pivot calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # P = (H + L + C) / 3
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    r4 = close_1d + (range_hl * 1.1 / 2)
    r3 = close_1d + (range_hl * 1.1 / 4)
    r2 = close_1d + (range_hl * 1.1 / 6)
    r1 = close_1d + (range_hl * 1.1 / 12)
    s1 = close_1d - (range_hl * 1.1 / 12)
    s2 = close_1d - (range_hl * 1.1 / 6)
    s3 = close_1d - (range_hl * 1.1 / 4)
    s4 = close_1d - (range_hl * 1.1 / 2)
    
    # Align pivot levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume filter: current volume > 20-period average (to avoid low-volume false signals)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    # Choppiness regime filter: use 4h data to avoid ranging markets
    # Calculate ATR(14) and ADX-like trend strength
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate directional movement for ADX
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness filter: avoid strong trends (ADX > 25) - we want range-bound markets
    trend_filter = adx_14 < 25
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Warmup for indicators
        # Skip if not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(adx_14[i]) or np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Range reversion logic: buy near support, sell near resistance
        # Long when price touches or goes below S1 and shows rejection
        # Short when price touches or goes above R1 and shows rejection
        
        # Check for rejection signals (price moves away from level after touching)
        touched_s1 = low[i] <= s1_aligned[i]
        touched_r1 = high[i] >= r1_aligned[i]
        
        # Rejection: price moves back toward pivot after touching level
        rejected_s1 = touched_s1 and close[i] > s1_aligned[i] * 1.005  # 0.5% bounce from S1
        rejected_r1 = touched_r1 and close[i] < r1_aligned[i] * 0.995  # 0.5% pullback from R1
        
        long_signal = rejected_s1 and volume_ok[i] and trend_filter[i]
        short_signal = rejected_r1 and volume_ok[i] and trend_filter[i]
        
        # Exit when price reaches opposite side or midpoint
        exit_long = close[i] >= pivot_aligned[i] or close[i] >= r1_aligned[i]
        exit_short = close[i] <= pivot_aligned[i] or close[i] <= s1_aligned[i]
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals