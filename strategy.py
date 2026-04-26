#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1wTrend_VolumeConfirm
Hypothesis: Ichimoku cloud twist (Senkou Span A/B cross) on 6h with 1w trend filter (price > 1w EMA50 for long, < for short) and volume confirmation (>1.5x average volume). 
The Kumo twist signals potential trend reversals with high probability. In bull markets: price above 1w EMA50, bullish Kumo twist (Senkou A crosses above Senkou B), and high volume → long. 
In bear markets: price below 1w EMA50, bearish Kumo twist (Senkou A crosses below Senkou B), and high volume → short. Uses discrete position sizing (0.25) to minimize fee churn. 
Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe. Uses 1w EMA50 for BTC/ETH edge via multi-timeframe alignment; avoids SOL-only bias by requiring 1w trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need warmup for Ichimoku (52 periods)
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2, plotted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used for signals to avoid look-ahead
    
    # Align Senkou Span A and B to LTF (they are already forward-shifted by 26 in calculation)
    # Since we calculate them on the same index, we need to shift them back 26 to align with price
    # But we want to use the values that were available 26 periods ago for current price
    # So we use the values without additional shift for current bar (they represent future cloud)
    # For signal detection, we compare current Senkou A/B (which is cloud 26 periods ahead)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 52 for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Need previous values to detect twist (cross)
        if i < 1:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Current Ichimoku values
        tenkan_curr = tenkan[i]
        kijun_curr = kijun[i]
        senkou_a_curr = senkou_a[i]
        senkou_b_curr = senkou_b[i]
        
        # Previous Ichimoku values for twist detection
        tenkan_prev = tenkan[i-1]
        kijun_prev = kijun[i-1]
        senkou_a_prev = senkou_a[i-1]
        senkou_b_prev = senkou_b[i-1]
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_1w_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(tenkan_curr) or np.isnan(kijun_curr) or np.isnan(senkou_a_curr) or 
            np.isnan(senkou_b_curr) or np.isnan(ema_val) or np.isnan(avg_vol)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Kumo twist detection: Senkou A crosses Senkou B
        bullish_twist = (senkou_a_prev <= senkou_b_prev) and (senkou_a_curr > senkou_b_curr)
        bearish_twist = (senkou_a_prev >= senkou_b_prev) and (senkou_a_curr < senkou_b_curr)
        
        # Long logic: price above 1w EMA50, bullish Kumo twist, and volume confirmation
        long_condition = (close_val > ema_val) and bullish_twist and volume_confirmed
        # Short logic: price below 1w EMA50, bearish Kumo twist, and volume confirmation
        short_condition = (close_val < ema_val) and bearish_twist and volume_confirmed
        
        # Exit logic: opposite Kumo twist or price crosses 1w EMA50
        exit_long = bearish_twist or (close_val < ema_val)
        exit_short = bullish_twist or (close_val > ema_val)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1wTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0