#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1wTrend_VolumeConfirm
Hypothesis: On 6h timeframe, Ichimoku Kumo Twist (Senkou Span A/B cross) with 1w trend filter (price >/ < 1w EMA50) and volume confirmation (>1.5x 20-bar avg) captures strong trend reversals in both bull and bear markets. Kumo Twist indicates momentum shift, 1w EMA ensures alignment with higher timeframe trend, volume confirms participation. Designed for 12-35 trades/year to minimize fee drag.
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
    
    # Get 1w data for HTF trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2
    
    # Kumo Twist: Senkou Span A crosses above/below Senkou Span B
    # We need previous values to detect cross
    senkou_a_prev = np.concatenate([[np.nan], senkou_a[:-1]])
    senkou_b_prev = np.concatenate([[np.nan], senkou_b[:-1]])
    
    # Kumo Twist Up: Senkou A crosses above Senkou B (prev A <= prev B and curr A > curr B)
    kumo_twist_up = (senkou_a_prev <= senkou_b_prev) & (senkou_a > senkou_b)
    # Kumo Twist Down: Senkou A crosses below Senkou B (prev A >= prev B and curr A < curr B)
    kumo_twist_down = (senkou_a_prev >= senkou_b_prev) & (senkou_a < senkou_b)
    
    # Align HTF trend to 6h
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume average (20-period) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(52, 20, 50)  # Senkou B, vol MA, EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(kumo_twist_up[i]) or 
            np.isnan(kumo_twist_down[i]) or 
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
        ema_val = ema_50_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = vol_val > 1.5 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Kumo Twist with trend and volume confirmation
            # Long: Kumo Twist Up with uptrend (close > EMA50) and volume confirmation
            long_signal = kumo_twist_up[i] and (close_val > ema_val) and volume_confirm
            # Short: Kumo Twist Down with downtrend (close < EMA50) and volume confirmation
            short_signal = kumo_twist_down[i] and (close_val < ema_val) and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Kumo Twist Down (exit long on momentum shift)
            if kumo_twist_down[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price crosses below 1w EMA50
            elif close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Kumo Twist Up (exit short on momentum shift)
            if kumo_twist_up[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price crosses above 1w EMA50
            elif close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1wTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0