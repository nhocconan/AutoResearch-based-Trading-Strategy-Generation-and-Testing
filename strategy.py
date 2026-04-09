#!/usr/bin/env python3
# 6h_ichimoku_cloud_breakout_v1
# Hypothesis: 6h strategy using Ichimoku cloud from 1d timeframe for trend filter and support/resistance, combined with 6h Tenkan/Kijun cross for entry timing. Uses volume confirmation (>1.3x 20-bar average) to filter false breakouts. Works in bull/bear: cloud acts as dynamic support/resistance, TK cross captures momentum shifts, volume confirms conviction. Discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_breakout_v1"
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
    
    # 6h Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_9 + low_9) / 2
    
    # 6h Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_26 + low_26) / 2
    
    # 6h Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # We'll handle the shift in alignment - for cloud calculation we need current values
    
    # 6h Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # 6h Chikou Span (Lagging Span): close shifted 26 periods behind
    # Not used for signals as it requires future data
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Multi-timeframe: 1d Ichimoku cloud for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Tenkan-sen (9-period)
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2
    
    # 1d Kijun-sen (26-period)
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26_1d + low_26_1d) / 2
    
    # 1d Senkou Span A
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # 1d Senkou Span B (52-period)
    high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (high_52_1d + low_52_1d) / 2
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate 6h cloud (Senkou Span A and B) - note: these are plotted 26 periods ahead
    # For current cloud, we use values calculated 26 periods ago
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # First 26 values will be invalid due to roll
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # 1d cloud for trend filter (also shifted)
    cloud_top_1d = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom_1d = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(cloud_top_1d[i]) or np.isnan(cloud_bottom_1d[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Price relative to 6h cloud
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        price_in_cloud = ~(price_above_cloud | price_below_cloud)
        
        # 6h TK cross
        tk_cross_bull = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_bear = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # 1d trend filter: price relative to 1d cloud
        trend_bull = close[i] > cloud_top_1d[i]
        trend_bear = close[i] < cloud_bottom_1d[i]
        
        if position == 1:  # Long position
            # Exit: price closes below 6h cloud OR bearish TK cross
            if price_below_cloud or tk_cross_bear:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 6h cloud OR bullish TK cross
            if price_above_cloud or tk_cross_bull:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: bullish TK cross + price above 6h cloud + volume + 1d bullish trend
            bullish_entry = tk_cross_bull and price_above_cloud and volume_confirmed and trend_bull
            # Enter short: bearish TK cross + price below 6h cloud + volume + 1d bearish trend
            bearish_entry = tk_cross_bear and price_below_cloud and volume_confirmed and trend_bear
            
            if bullish_entry:
                position = 1
                signals[i] = 0.25
            elif bearish_entry:
                position = -1
                signals[i] = -0.25
    
    return signals