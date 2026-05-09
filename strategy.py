#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_IchimokuCloud_1dTrend_Filter"
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
    
    # Get 1d data for trend filter and Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 days for Ichimoku
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components on 1d data
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    high_9 = pd.Series(df_1d['high'].values).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low'].values).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    high_26 = pd.Series(df_1d['high'].values).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low'].values).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # We'll align this later; for cloud calculation we need current values
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    high_52 = pd.Series(df_1d['high'].values).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low'].values).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Volume filter: above 1.3x 20-period average (20*6h = 120h ~ 5 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 20)  # Wait for Ichimoku and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(ema_50_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.3 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 8-20)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        in_session = 8 <= hour <= 20
        
        # Ichimoku signals
        # Tenkan/Kijun cross
        tk_cross = tenkan_6h[i] > kijun_6h[i]
        tk_cross_prev = tenkan_6h[i-1] <= kijun_6h[i-1] if i > 0 else False
        
        # Price above/below cloud
        cloud_top = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        if position == 0:
            # Long: TK cross up + price above cloud + 1d uptrend + volume + session
            if (tk_cross and tk_cross_prev and  # Fresh TK cross up
                price_above_cloud and
                close[i] > ema_50_6h[i] and     # 1d uptrend filter
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + price below cloud + 1d downtrend + volume + session
            elif ((not tk_cross) and (not tk_cross_prev) and  # Fresh TK cross down
                  price_below_cloud and
                  close[i] < ema_50_6h[i] and     # 1d downtrend filter
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TK cross down OR price breaks below cloud
            if ((not tk_cross) and (not tk_cross_prev)) or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross up OR price breaks above cloud
            if (tk_cross and tk_cross_prev) or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals