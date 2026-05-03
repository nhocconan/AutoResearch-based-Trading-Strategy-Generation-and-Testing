#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-week Ichimoku cloud filter with 1-day TK cross for entry timing and volume confirmation.
# Uses Ichimoku from weekly timeframe for major trend/cloud filter (bullish when price > cloud, bearish when price < cloud),
# 1-day Tenkan/Kijun cross for entry signals, and 6h volume spike for confirmation.
# Designed to work in both bull and bear markets by only taking trades in direction of weekly Ichimoku trend.
# Targets 12-35 trades/year by requiring confluence of weekly trend, daily momentum, and volume spike.

name = "6h_Ichimoku_1wCloud_1dTKCross_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for TK cross
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # Get 1w data for Ichimoku cloud
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Calculate 1d Tenkan-sen (9-period) and Kijun-sen (26-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen: (9-period high + 9-period low) / 2
    tenkan_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    tenkan_1d_vals = tenkan_1d.values
    
    # Kijun-sen: (26-period high + 26-period low) / 2
    kijun_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    kijun_1d_vals = kijun_1d.values
    
    # Calculate 1w Ichimoku components
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Tenkan-sen (9-period) for 1w
    tenkan_1w = (pd.Series(high_1w).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low_1w).rolling(window=9, min_periods=9).min()) / 2
    tenkan_1w_vals = tenkan_1w.values
    
    # Kijun-sen (26-period) for 1w
    kijun_1w = (pd.Series(high_1w).rolling(window=26, min_periods=26).max() + 
                pd.Series(low_1w).rolling(window=26, min_periods=26).min()) / 2
    kijun_1w_vals = kijun_1w.values
    
    # Senkou Span A: (Tenkan + Kijun) / 2 plotted 26 periods ahead
    senkou_a_1w = ((tenkan_1w_vals + kijun_1w_vals) / 2)
    
    # Senkou Span B: (52-period high + 52-period low) / 2 plotted 26 periods ahead
    senkou_b_1w = (pd.Series(high_1w).rolling(window=52, min_periods=52).max() + 
                   pd.Series(low_1w).rolling(window=52, min_periods=52).min()) / 2
    senkou_b_1w_vals = senkou_b_1w.values
    
    # Align 1d TK cross values to 6h
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d_vals)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d_vals)
    
    # Align 1w Ichimoku cloud to 6h (Senkou Span A and B)
    # Cloud is plotted 26 periods ahead, so we need to align with additional_delay_bars=26
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w, additional_delay_bars=26)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w_vals, additional_delay_bars=26)
    
    # Calculate 6h volume spike: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        tenkan_1d_val = tenkan_1d_aligned[i]
        kijun_1d_val = kijun_1d_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if (np.isnan(tenkan_1d_val) or np.isnan(kijun_1d_val) or 
            np.isnan(senkou_a_val) or np.isnan(senkou_b_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine Ichimoku cloud trend from 1w
        # Bullish cloud: Senkou A > Senkou B
        # Bearish cloud: Senkou A < Senkou B
        bullish_cloud = senkou_a_val > senkou_b_val
        bearish_cloud = senkou_a_val < senkou_b_val
        
        # TK cross signals from 1d
        tk_bullish_cross = tenkan_1d_val > kijun_1d_val
        tk_bearish_cross = tenkan_1d_val < kijun_1d_val
        
        # Entry conditions with volume confirmation
        # Long: bullish cloud + TK bullish cross + volume spike
        long_entry = bullish_cloud and tk_bullish_cross and vol_spike
        # Short: bearish cloud + TK bearish cross + volume spike
        short_entry = bearish_cloud and tk_bearish_cross and vol_spike
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on TK bearish cross (momentum change)
            if tk_bearish_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on TK bullish cross (momentum change)
            if tk_bullish_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals