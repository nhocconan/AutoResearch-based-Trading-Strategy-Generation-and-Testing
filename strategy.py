#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud (TK cross + cloud filter from 1d) with volume confirmation.
# Long when Tenkan-sen crosses above Kijun-sen, price above cloud, and volume > 1.5x 20-bar avg.
# Short when Tenkan-sen crosses below Kijun-sen, price below cloud, and volume > 1.5x 20-bar avg.
# Exit on opposite TK cross or when price crosses the cloud midpoint (Kumo).
# Uses proven Ichimoku structure with 1d HTF for cloud calculation (more stable) and volume confirmation.
# 6h timeframe targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Ichimoku works in both bull/bear markets: trend following via cloud, momentum via TK cross.

name = "6h_Ichimoku_TK_Cross_1dCloud_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 26*2 for Ichimoku
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    
    # Align Ichimoku components to 6h timeframe (completed 1d bar only)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Kumo (cloud) top and bottom
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Kumo midpoint (for exit)
    kumo_mid = (cloud_top + cloud_bottom) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average (balanced to avoid overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for Ichimoku (52+26)
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(kumo_mid[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_tenkan = tenkan_sen_aligned[i]
        curr_kijun = kijun_sen_aligned[i]
        curr_cloud_top = cloud_top[i]
        curr_cloud_bottom = cloud_bottom[i]
        curr_kumo_mid = kumo_mid[i]
        curr_volume_confirm = volume_confirm[i]
        
        # TK cross signals
        bullish_tk_cross = curr_tenkan > curr_kijun
        bearish_tk_cross = curr_tenkan < curr_kijun
        
        # Price relative to cloud
        price_above_cloud = curr_close > curr_cloud_top
        price_below_cloud = curr_close < curr_cloud_bottom
        price_in_cloud = (curr_close >= curr_cloud_bottom) & (curr_close <= curr_cloud_top)
        
        if position == 0:  # Flat - look for new entries
            # Long: bullish TK cross, price above cloud, volume confirmation
            if (bullish_tk_cross and price_above_cloud and curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross, price below cloud, volume confirmation
            elif (bearish_tk_cross and price_below_cloud and curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit: bearish TK cross OR price crosses below cloud midpoint (mean reversion)
            if (bearish_tk_cross or curr_close <= curr_kumo_mid):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: bullish TK cross OR price crosses above cloud midpoint (mean reversion)
            if (bullish_tk_cross or curr_close >= curr_kumo_mid):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals