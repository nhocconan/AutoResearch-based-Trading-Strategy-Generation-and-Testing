#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Ichimoku Cloud (TK cross + price above/below cloud) with 1d volume spike filter.
# Long when Tenkan > Kijun (bullish TK cross), price above Senkou Span A/B (above cloud), and volume > 1.5x 20-period average.
# Short when Tenkan < Kijun (bearish TK cross), price below Senkou Span A/B (below cloud), and volume > 1.5x 20-period average.
# Exit on opposite TK cross or price re-enters the cloud.
# Designed to capture trending moves with Ichimoku's trend/momentum/cloud filter, effective in both bull and bear markets.
# Target: 60-120 total trades over 4 years (15-30/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Ichimoku Components ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
              pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
             pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b.values)
    
    # === 1d Volume Spike Filter ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all Ichimoku data is valid (max 52+26=78 periods)
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(volume_spike[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        
        # Ichimoku conditions
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        
        # Cloud boundaries (top and bottom of cloud)
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # TK cross conditions
        tk_bullish = tenkan_val > kijun_val
        tk_bearish = tenkan_val < kijun_val
        
        # Price relative to cloud
        price_above_cloud = price > cloud_top
        price_below_cloud = price < cloud_bottom
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit on bearish TK cross or price re-enters cloud
            if tk_bearish or (price <= cloud_top and price >= cloud_bottom):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit on bullish TK cross or price re-enters cloud
            if tk_bullish or (price <= cloud_top and price >= cloud_bottom):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bullish TK cross, price above cloud, volume spike
            if tk_bullish and price_above_cloud and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bearish TK cross, price below cloud, volume spike
            elif tk_bearish and price_below_cloud and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_IchimokuTK_Cross_CloudFilter_1dVolumeSpike_V1"
timeframe = "6h"
leverage = 1.0