# 6h Ichimoku Cloud Breakout with 1d Trend Filter
# Combines Ichimoku (Tenkan/Kijun cross + cloud filter) from 6h with 1d trend filter.
# Long when: Tenkan > Kijun, price above cloud, and 1d close > 1d EMA50.
# Short when: Tenkan < Kijun, price below cloud, and 1d close < 1d EMA50.
# Exit when Tenkan/Kijun cross reverses or price re-enters cloud.
# Designed for low trade frequency (~15-35/year) with strong edge in both bull and bear markets.
# Ichimoku captures momentum and support/resistance; 1d EMA50 filters trend direction.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Cloud boundaries: Senkou Span A and B
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        upper_cloud_val = upper_cloud[i]
        lower_cloud_val = lower_cloud[i]
        ema_val = ema_50_aligned[i]
        
        if position == 0:
            # Long conditions: Tenkan > Kijun, price above cloud, and 1d uptrend
            if tenkan_val > kijun_val and price > upper_cloud_val and close[i] > ema_val:
                signals[i] = 0.25
                position = 1
            # Short conditions: Tenkan < Kijun, price below cloud, and 1d downtrend
            elif tenkan_val < kijun_val and price < lower_cloud_val and close[i] < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Tenkan/Kijun cross reverses or price re-enters cloud
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Tenkan < Kijun or price re-enters cloud
                if tenkan_val < kijun_val or price < upper_cloud_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Tenkan > Kijun or price re-enters cloud
                if tenkan_val > kijun_val or price > lower_cloud_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0