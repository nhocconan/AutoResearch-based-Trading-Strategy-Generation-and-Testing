#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_WeeklyTrend_Confirm
Hypothesis: 6h Ichimoku Kumo twist (Senkou Span A/B cross) with 1w EMA50 trend filter.
Long when Senkou A crosses above Senkou B with price above cloud and 1w EMA50 uptrend.
Short when Senkou A crosses below Senkou B with price below cloud and 1w EMA50 downtrend.
Exit on opposite Kumo cross or trend reversal.
Uses discrete sizing (0.25) to minimize fee churn. Target: 12-37 trades/year on 6h.
Works in bull via trend-following Kumo breaks, in bear via cloud rejection and trend alignment.
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
    
    # Get 6h data for Ichimoku calculations (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:  # need 52 for Senkou B
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current price vs Kumo (cloud) - we need to align properly
    # For cloud twist signal, we compare current Senkou A and B (already shifted)
    # But for price vs cloud, we need current Senkou values (which are plotted 26 periods ahead)
    # So current cloud is Senkou values from 26 periods ago
    senkou_a_current = np.roll(senkou_a, 26)
    senkou_b_current = np.roll(senkou_b, 26)
    senkou_a_current[:26] = np.nan
    senkou_b_current[:26] = np.nan
    
    # Align Ichimoku components to original timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)
    senkou_a_current_aligned = align_htf_to_ltf(prices, df_6h, senkou_a_current)
    senkou_b_current_aligned = align_htf_to_ltf(prices, df_6h, senkou_b_current)
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(senkou_a_current_aligned[i]) or np.isnan(senkou_b_current_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Kumo twist: Senkou A cross Senkou B
        # We need previous and current values to detect cross
        if i == start_idx:
            prev_senkou_a = senkou_a_aligned[i-1]
            prev_senkou_b = senkou_b_aligned[i-1]
        else:
            prev_senkou_a = senkou_a_aligned[i-1]
            prev_senkou_b = senkou_b_aligned[i-1]
        
        curr_senkou_a = senkou_a_aligned[i]
        curr_senkou_b = senkou_b_aligned[i]
        
        # Bullish twist: Senkou A crosses above Senkou B
        bullish_twist = (prev_senkou_a <= prev_senkou_b) and (curr_senkou_a > curr_senkou_b)
        # Bearish twist: Senkou A crosses below Senkou B
        bearish_twist = (prev_senkou_a >= prev_senkou_b) and (curr_senkou_a < curr_senkou_b)
        
        # Price relative to cloud (current Senkou values)
        price_above_cloud = (close[i] > senkou_a_current_aligned[i]) and (close[i] > senkou_b_current_aligned[i])
        price_below_cloud = (close[i] < senkou_a_current_aligned[i]) and (close[i] < senkou_b_current_aligned[i])
        
        if position == 0:
            # Long: bullish twist + price above cloud + 1w uptrend
            long_signal = bullish_twist and price_above_cloud and (close[i] > ema_50_1w_aligned[i])
            # Short: bearish twist + price below cloud + 1w downtrend
            short_signal = bearish_twist and price_below_cloud and (close[i] < ema_50_1w_aligned[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: bearish twist or price below cloud or trend reverses
            exit_signal = bearish_twist or (close[i] < senkou_a_current_aligned[i]) or (close[i] < ema_50_1w_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: bullish twist or price above cloud or trend reverses
            exit_signal = bullish_twist or (close[i] > senkou_a_current_aligned[i]) or (close[i] > ema_50_1w_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_WeeklyTrend_Confirm"
timeframe = "6h"
leverage = 1.0