#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_VolumeConfirm
Hypothesis: On 6h timeframe, Ichimoku Kumo Twist (Senkou Span A/B cross) with 1d trend filter (price > 1d EMA50 for long, < for short) and volume confirmation (>1.5x 20-bar avg) captures strong trend reversals with low trade frequency. Kumo Twist indicates potential trend change, 1d EMA50 ensures alignment with higher timeframe momentum, and volume confirms participation. Designed for 12-30 trades/year to minimize fee drag. Works in both bull and bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend and Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Kumo Twist occurs when Senkou Span A crosses Senkou Span B
    # We detect the cross and use it for signals
    # For alignment, we need the values at the time of the cross (not shifted)
    # So we use Senkou Span A and B without the forward shift for twist detection
    senkou_a_current = (tenkan_sen + kijun_sen) / 2
    senkou_b_current = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_current)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_current)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(100, 20)  # Ichimoku needs 52 periods, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        tenkan_val = tenkan_sen_aligned[i]
        kijun_val = kijun_sen_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        
        # Kumo Twist: Senkou Span A crosses Senkou Span B
        # Bullish twist: Senkou A crosses above Senkou B
        # Bearish twist: Senkou A crosses below Senkou B
        # We need previous values to detect cross
        if i > start_idx:
            prev_senkou_a = senkou_a_aligned[i-1]
            prev_senkou_b = senkou_b_aligned[i-1]
            bullish_twist = (prev_senkou_a <= prev_senkou_b) and (senkou_a_val > senkou_b_val)
            bearish_twist = (prev_senkou_a >= prev_senkou_b) and (senkou_a_val < senkou_b_val)
        else:
            bullish_twist = False
            bearish_twist = False
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = vol_val > 1.5 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Kumo Twist with trend and volume confirmation
            # Long: bullish twist with price > 1d EMA50 and volume confirmation
            long_signal = bullish_twist and (close_val > ema_val) and volume_confirm
            # Short: bearish twist with price < 1d EMA50 and volume confirmation
            short_signal = bearish_twist and (close_val < ema_val) and volume_confirm
            
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
            # Exit conditions:
            # 1. Kumo Twist in opposite direction (bearish twist)
            # 2. Price crosses below 1d EMA50 (trend change)
            if bearish_twist or (close_val < ema_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Kumo Twist in opposite direction (bullish twist)
            # 2. Price crosses above 1d EMA50 (trend change)
            if bullish_twist or (close_val > ema_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0