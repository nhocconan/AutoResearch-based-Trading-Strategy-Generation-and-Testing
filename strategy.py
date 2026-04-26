#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_12hTrend_VolumeConfirm
Hypothesis: Ichimoku TK cross on 6h with cloud filter from 1d and 12h EMA trend confirmation + volume spike.
In bull markets: price above cloud, TK bullish cross, 12h uptrend, and volume >1.5x average → long.
In bear markets: price below cloud, TK bearish cross, 12h downtrend, and volume >1.5x average → short.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 trades over 4 years (12-37/year) on 6h timeframe.
Ichimoku cloud provides dynamic support/resistance that adapts to volatility, working in both trending and ranging markets.
The 12h trend filter ensures we trade with the intermediate-term trend, reducing whipsaws.
Volume confirmation ensures breakouts have conviction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for Ichimoku calculation
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF cloud calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 for Senkou Span B
        return np.zeros(n)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:  # Need 21 for EMA
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_52 + min_low_52) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not needed for signals)
    
    # Align Ichimoku components to 6h timeframe (already shifted for cloud)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate 12h EMA21 for trend filter
    ema_21_12h = pd.Series(df_12h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 52 for Senkou B, 26 for alignment shift)
    start_idx = 52 + 26  # 78 periods to ensure all Ichimoku components are valid
    
    for i in range(start_idx, n):
        # Get current Ichimoku values
        tenkan = tenkan_aligned[i]
        kijun = kijun_aligned[i]
        senkou_a = senkou_a_aligned[i]
        senkou_b = senkou_b_aligned[i]
        ema_val = ema_21_12h_aligned[i]
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Skip if any data not ready
        if (np.isnan(tenkan) or np.isnan(kijun) or np.isnan(senkou_a) or 
            np.isnan(senkou_b) or np.isnan(ema_val) or np.isnan(avg_vol)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = max(senkou_a, senkou_b)
        lower_cloud = min(senkou_a, senkou_b)
        
        # TK Cross signals
        tk_bullish = tenkan > kijun
        tk_bearish = tenkan < kijun
        
        # Price relative to cloud
        price_above_cloud = close_val > upper_cloud
        price_below_cloud = close_val < lower_cloud
        price_in_cloud = (close_val >= lower_cloud) and (close_val <= upper_cloud)
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Long logic: price above cloud, TK bullish cross, 12h uptrend, and volume confirmation
        long_condition = price_above_cloud and tk_bullish and (close_val > ema_val) and volume_confirmed
        # Short logic: price below cloud, TK bearish cross, 12h downtrend, and volume confirmation
        short_condition = price_below_cloud and tk_bearish and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: TK cross reversal or price re-enters cloud
        exit_long = (not tk_bullish) or price_in_cloud or price_below_cloud
        exit_short = (not tk_bearish) or price_in_cloud or price_above_cloud
        
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

name = "6h_Ichimoku_Cloud_Breakout_12hTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0