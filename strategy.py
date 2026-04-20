#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# - Calculate Ichimoku components (Tenkan, Kijun, Senkou A/B) from 6h data
# - Use 1d EMA50 as trend filter: only long when price > EMA50, short when price < EMA50
# - Enter long when Tenkan crosses above Kijun AND price above cloud (Senkou span)
# - Enter short when Tenkan crosses below Kijun AND price below cloud
# - Require volume > 1.5x 20-period average for confirmation
# - Exit when Tenkan/Kijun cross reverses or price crosses opposite Senkou span
# - Target: 15-30 trades per year per symbol (60-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 6h data for Ichimoku calculation
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after warmup for Senkou B
        # Skip if NaN in critical values
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or \
           np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Determine cloud boundaries (Senkou A and B)
        upper_cloud = np.maximum(senkou_a[i], senkou_b[i])
        lower_cloud = np.minimum(senkou_a[i], senkou_b[i])
        
        if position == 0:
            # Long entry: Tenkan crosses above Kijun, price above cloud, above 1d EMA50, volume surge
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] and  # Cross up
                price > upper_cloud and 
                price > ema_50_1d_aligned[i] and
                vol > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Tenkan crosses below Kijun, price below cloud, below 1d EMA50, volume surge
            elif (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] and  # Cross down
                  price < lower_cloud and 
                  price < ema_50_1d_aligned[i] and
                  vol > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun OR price drops below cloud
            if (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]) or price < lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun OR price rises above cloud
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]) or price > upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_1dEMA50_VolumeFilter"
timeframe = "6h"
leverage = 1.0