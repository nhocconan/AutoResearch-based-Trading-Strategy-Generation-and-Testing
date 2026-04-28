#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # Ichimoku Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(df_1d['high'].values).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low'].values).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Ichimoku Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(df_1d['high'].values).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low'].values).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Ichimoku Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Ichimoku Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(df_1d['high'].values).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low'].values).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Align to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Ichimoku TK Cross signals
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i]
        
        # Cloud (Kumo) conditions
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Entry conditions: 
        # Long: TK cross bullish + price above cloud
        # Short: TK cross bearish + price below cloud
        long_entry = tk_cross_up and vol_filter and price_above_cloud
        short_entry = tk_cross_down and vol_filter and price_below_cloud
        
        # Exit conditions: opposite TK cross
        long_exit = tk_cross_down and position == 1
        short_exit = tk_cross_up and position == -1
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_TKCross_CloudFilter_Volume_Session"
timeframe = "6h"
leverage = 1.0