#!/usr/bin/env python3
"""
6h Ichimoku Cloud with 1d Filter and Volume Confirmation
Hypothesis: Ichimoku TK cross acts as momentum signal, filtered by 1d cloud color (trend) and volume spikes.
Works in bull markets via TK cross above cloud, bear markets via TK cross below cloud.
Volume confirmation ensures only high-conviction signals trigger entries.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_1d_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components (9, 26, 52)
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
    
    # 1d data for trend filter (cloud color)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Ichimoku components
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2
    
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26_1d + low_26_1d) / 2
    
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (high_52_1d + low_52_1d) / 2
    
    # 1d cloud color: green if Senkou A > Senkou B (uptrend), red otherwise
    cloud_green_1d = senkou_a_1d > senkou_b_1d
    cloud_red_1d = senkou_a_1d < senkou_b_1d
    
    cloud_green_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_green_1d.astype(float))
    cloud_red_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_red_1d.astype(float))
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(cloud_green_1d_aligned[i]) or np.isnan(cloud_red_1d_aligned[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: TK cross down (Tenkan < Kijun) OR price enters cloud
            if (tenkan[i] < kijun[i] or 
                (close[i] > senkou_a[i] and close[i] < senkou_b[i]) or
                (close[i] < senkou_a[i] and close[i] > senkou_b[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TK cross up (Tenkan > Kijun) OR price enters cloud
            if (tenkan[i] > kijun[i] or 
                (close[i] > senkou_a[i] and close[i] < senkou_b[i]) or
                (close[i] < senkou_a[i] and close[i] > senkou_b[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Bullish TK cross: Tenkan crosses above Kijun
            tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
            # Bearish TK cross: Tenkan crosses below Kijun
            tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
            
            # Price above/below cloud
            price_above_cloud = close[i] > max(senkou_a[i], senkou_b[i])
            price_below_cloud = close[i] < min(senkou_a[i], senkou_b[i])
            
            # Long: bullish TK cross + price above cloud + 1d uptrend (green cloud) + volume spike
            if (tk_cross_up and price_above_cloud and 
                cloud_green_1d_aligned[i] > 0.5 and vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: bearish TK cross + price below cloud + 1d downtrend (red cloud) + volume spike
            elif (tk_cross_down and price_below_cloud and 
                  cloud_red_1d_aligned[i] > 0.5 and vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals