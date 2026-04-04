#!/usr/bin/env python3
"""
exp_6575_6h_ichimoku_cloud_1w_trend_v1
Hypothesis: 6h Ichimoku cloud breakout with 1w trend filter. Uses Ichimoku (Tenkan/Kijun/Senkou) for entry timing and 1w cloud color for trend direction. Works in bull/bear by only taking trades aligned with weekly trend (cloud color). Targets 50-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6575_6h_ichimoku_cloud_1w_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_PERIOD = 52
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
MAX_HOLD_BARS = 40  # ~10 days max hold

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w Ichimoku components for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_1w = (pd.Series(high_1w).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                 pd.Series(low_1w).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    # 1w Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_1w = (pd.Series(high_1w).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                pd.Series(low_1w).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    # 1w Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1w = ((tenkan_1w + kijun_1w) / 2)
    # 1w Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b_1w = ((pd.Series(high_1w).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).max() + 
                    pd.Series(low_1w).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).min()) / 2)
    
    # 1w Cloud color: green (bullish) when Senkou A > Senkou B, red (bearish) when A < B
    cloud_green_1w = senkou_a_1w > senkou_b_1w
    cloud_red_1w = senkou_a_1w < senkou_b_1w
    
    # Align 1w Ichimoku components to 6h (with shift(1) for completed weekly bars only)
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w.values)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w.values)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w.values)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w.values)
    cloud_green_1w_aligned = align_htf_to_ltf(prices, df_1w, cloud_green_1w.astype(float))
    cloud_red_1w_aligned = align_htf_to_ltf(prices, df_1w, cloud_red_1w.astype(float))
    
    # Calculate 6h Ichimoku components for entry signals
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Tenkan-sen (Conversion Line)
    tenkan_6h = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                 pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    # 6h Kijun-sen (Base Line)
    kijun_6h = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    # 6h Senkou Span A (Leading Span A)
    senkou_a_6h = ((tenkan_6h + kijun_6h) / 2)
    # 6h Senkou Span B (Leading Span B)
    senkou_b_6h = ((pd.Series(high).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).max() + 
                    pd.Series(low).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).min()) / 2)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period (need enough data for 52-period indicators)
    start = max(SENKOU_PERIOD, KIJUN_PERIOD, VOL_MA_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or 
            np.isnan(senkou_a_1w_aligned[i]) or np.isnan(senkou_b_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Determine 1w trend bias from cloud color
        # Green cloud = bullish bias, Red cloud = bearish bias
        bullish_bias = cloud_green_1w_aligned[i] > 0.5
        bearish_bias = cloud_red_1w_aligned[i] > 0.5
        
        # 6h Ichimoku signals
        # Tenkan/Kijun cross: Tenkan > Kijun = bullish, Tenkan < Kijun = bearish
        tenkan_above_kijun = tenkan_6h[i] > kijun_6h[i]
        tenkan_below_kijun = tenkan_6h[i] < kijun_6h[i]
        
        # Price relative to cloud: price above cloud = bullish, below cloud = bearish
        price_above_cloud = close[i] > max(senkou_a_6h[i], senkou_b_6h[i])
        price_below_cloud = close[i] < min(senkou_a_6h[i], senkou_b_6h[i])
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions: Tenkan/Kijun cross reversal OR price re-enters cloud
        if position == 1:  # long position
            # Exit if Tenkan crosses below Kijun OR price re-enters cloud
            exit_long = tenkan_below_kijun or not price_above_cloud
            # Time-based exit
            exit_long = exit_long or bars_since_entry >= MAX_HOLD_BARS
            if exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            # Exit if Tenkan crosses above Kijun OR price re-enters cloud
            exit_short = tenkan_above_kijun or not price_below_cloud
            # Time-based exit
            exit_short = exit_short or bars_since_entry >= MAX_HOLD_BARS
            if exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            # Long: bullish 6h signal + bullish 1w bias + volume
            if (tenkan_above_kijun and price_above_cloud) and bullish_bias and vol_confirm:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            # Short: bearish 6h signal + bearish 1w bias + volume
            elif (tenkan_below_kijun and price_below_cloud) and bearish_bias and vol_confirm:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals