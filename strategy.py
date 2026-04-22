# 6h_Pivot_Fade_Strategy_v1
# Strategy Type: Pivot Point Fade with Trend Filter
# Timeframe: 6h
# Hypothesis: Fade at daily pivot support/resistance levels (S1/R1) during ranging markets (low ADX),
# but follow breakouts beyond S2/R2 during trending markets (high ADX). Works in both bull/bear
# by adapting to regime - mean reversion in range, momentum in trend. Uses 1d ADX for regime filter.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for pivot points and ADX (regime filter)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points (standard formula)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Calculate ADX for regime detection (trending vs ranging)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM (14-period)
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    adx_6h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Price array
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(adx_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        pivot_val = pivot_6h[i]
        r1_val = r1_6h[i]
        s1_val = s1_6h[i]
        r2_val = r2_6h[i]
        s2_val = s2_6h[i]
        adx_val = adx_6h[i]
        
        # Volume filter: avoid extremely low volume periods
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = vol > 0.5 * vol_ma_20  # at least half average volume
        
        if position == 0:
            # Regime-based logic:
            # Ranging market (ADX < 25): fade at S1/R1 (mean reversion)
            # Trending market (ADX >= 25): breakout beyond S2/R2 (momentum)
            
            if adx_val < 25:  # Ranging market - mean reversion
                # Long: price at or below S1 with rejection (close > open) and volume
                if price <= s1_val and close[i] > prices['open'].iloc[i] and vol_filter:
                    signals[i] = 0.25
                    position = 1
                # Short: price at or above R1 with rejection (close < open) and volume
                elif price >= r1_val and close[i] < prices['open'].iloc[i] and vol_filter:
                    signals[i] = -0.25
                    position = -1
            else:  # Trending market - momentum breakout
                # Long: price breaks above R2 with volume
                if price > r2_val and vol_filter:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below S2 with volume
                elif price < s2_val and vol_filter:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on: 
                # 1. Price reaches R1 (take profit in range) or S2 (stop loss)
                # 2. In trend: price fails to hold above S1
                if adx_val < 25:  # ranging
                    if price >= r1_val or price <= s2_val:
                        exit_signal = True
                else:  # trending
                    if price <= s1_val:  # break of short-term support
                        exit_signal = True
            
            elif position == -1:  # short position
                # Exit on:
                # 1. Price reaches S1 (take profit in range) or R2 (stop loss)
                # 2. In trend: price fails to hold below R1
                if adx_val < 25:  # ranging
                    if price <= s1_val or price >= r2_val:
                        exit_signal = True
                else:  # trending
                    if price >= r1_val:  # break of short-term resistance
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Pivot_Fade_Strategy_v1"
timeframe = "6h"
leverage = 1.0