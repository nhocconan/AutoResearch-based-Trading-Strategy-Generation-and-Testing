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
    
    # Get 12h data for Ichimoku components
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Ichimoku on 12h (9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods
    high_52 = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Align Ichimoku components to 6h
    tenkan_6h = align_htf_to_ltf(prices, df_12h, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_12h, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_12h, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_12h, senkou_b)
    
    # Calculate 12h ATR for volatility filter
    tr1 = pd.Series(high_12h).rolling(window=1).max() - pd.Series(low_12h).rolling(window=1).min()
    tr2 = abs(pd.Series(high_12h).rolling(window=1).max() - pd.Series(close_12h).shift(1))
    tr3 = abs(pd.Series(low_12h).rolling(window=1).min() - pd.Series(close_12h).shift(1))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr_12h.rolling(window=14, min_periods=14).mean().values
    atr_12h_6h = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate volume moving average (20-period on 6h)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26, 20)  # need Ichimoku components, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or 
            np.isnan(atr_12h_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_12h_6h[i] > 0.5 * np.nanmedian(atr_12h_6h[max(0, i-100):i+1])
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Cloud relationship: price above/below cloud
        cloud_top = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        if position == 0:
            # Long entry: TK cross bullish, price above cloud, with volume and volatility
            if (tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1] and
                price_above_cloud and vol_confirmed and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short entry: TK cross bearish, price below cloud, with volume and volatility
            elif (tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1] and
                  price_below_cloud and vol_confirmed and vol_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: TK cross bearish or price drops below cloud
            if (tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]) or \
               close[i] < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TK cross bullish or price rises above cloud
            if (tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]) or \
               close[i] > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter"
timeframe = "6h"
leverage = 1.0