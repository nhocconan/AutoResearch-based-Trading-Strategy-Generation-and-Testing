#!/usr/bin/env python3
name = "6h_Ichimoku_Cloud_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Load daily data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52)
    tenkan_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max()
    tenkan_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    kijun_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max()
    kijun_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    senkou_b_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max()
    senkou_b_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min()
    senkou_b = ((senkou_b_high + senkou_b_low) / 2).shift(26)
    
    # Convert to numpy arrays
    tenkan_sen_arr = tenkan_sen.values
    kijun_sen_arr = kijun_sen.values
    senkou_a_arr = senkou_a.values
    senkou_b_arr = senkou_b.values
    
    # Align to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_arr)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_arr)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_arr)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_arr)
    
    # Weekly trend filter: price above/below weekly EMA50
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Ichimoku signals
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        senkou_a = senkou_a_aligned[i]
        senkou_b = senkou_b_aligned[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        # Bullish: Tenkan > Kijun and price above cloud
        # Bearish: Tenkan < Kijun and price below cloud
        bullish_setup = tenkan > kijun and close[i] > cloud_top
        bearish_setup = tenkan < kijun and close[i] < cloud_bottom
        
        if position == 0:
            # Long: bullish setup + above weekly EMA50 + volume filter
            if bullish_setup and close[i] > ema_50_1w_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish setup + below weekly EMA50 + volume filter
            elif bearish_setup and close[i] < ema_50_1w_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish setup or price below weekly EMA50
            if bearish_setup or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish setup or price above weekly EMA50
            if bullish_setup or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals