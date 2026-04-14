#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Ichimoku Cloud with 1-day volume confirmation
# Long when price above Kumo (cloud), Tenkan > Kijun, and volume > 1.5x 20-period average
# Short when price below Kumo, Tenkan < Kijun, and volume > 1.5x 20-period average
# Exit when price crosses into the Kumo (opposite side)
# Ichimoku provides multi-line trend confirmation, Kumo acts as dynamic support/resistance
# Volume filter ensures institutional participation. Target: 25-40 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku Cloud calculations (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high9 + low9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high26 + low26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high52 + low52) / 2
    
    # Kumo (Cloud) boundaries: Senkou Span A and B shifted 26 periods ahead
    # For look-ahead avoidance, we use the cloud values from 26 periods ago
    # which represent the cloud's current position
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # First 26 values are invalid (rolled from end)
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (52 for Senkou B + 26 shift buffer)
    start = 78
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_shifted[i]) or np.isnan(senkou_b_shifted[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        # Determine cloud boundaries (Senkou A and B)
        upper_cloud = max(senkou_a_shifted[i], senkou_b_shifted[i])
        lower_cloud = min(senkou_a_shifted[i], senkou_b_shifted[i])
        
        if position == 0:
            # Long setup: price above cloud, Tenkan > Kijun, volume confirmation
            if (price > upper_cloud and tenkan[i] > kijun[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price below cloud, Tenkan < Kijun, volume confirmation
            elif (price < lower_cloud and tenkan[i] < kijun[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses into cloud (below upper cloud boundary)
            if price < upper_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses into cloud (above lower cloud boundary)
            if price > lower_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Ichimoku_Cloud_Volume"
timeframe = "4h"
leverage = 1.0