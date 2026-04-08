#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ichimoku_cloud_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Ichimoku components (based on 1d data)
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2.0
    
    # Align Ichimoku components to 4h timeframe (with shift(1) for completed bars)
    tenkan_4h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_4h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_4h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_4h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_4h, senkou_b_4h)
    cloud_bottom = np.minimum(senkou_a_4h, senkou_b_4h)
    
    # 4h trend filter: 50-period EMA
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_4h[i]) or np.isnan(kijun_4h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(ema_50[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price below cloud or trend fails
            if close[i] < cloud_bottom[i] or close[i] < ema_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above cloud or trend fails
            if close[i] > cloud_top[i] or close[i] > ema_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter
            bullish = close[i] > ema_50[i]
            bearish = close[i] < ema_50[i]
            
            # Long: price above cloud + bullish trend + volume
            if (close[i] > cloud_top[i] and 
                bullish and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price below cloud + bearish trend + volume
            elif (close[i] < cloud_bottom[i] and 
                  bearish and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals