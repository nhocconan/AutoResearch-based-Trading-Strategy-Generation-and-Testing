#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Ichimoku_TK_Cross_Cloud_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Ichimoku components: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (52)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_6h, senkou_span_b_6h)
    cloud_bottom = np.minimum(senkou_span_a_6h, senkou_span_b_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # warmup for Senkou Span B
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: TK cross bullish (Tenkan > Kijun) AND price above cloud
            tk_cross_bullish = tenkan_sen_6h[i] > kijun_sen_6h[i]
            price_above_cloud = close[i] > cloud_top[i]
            
            # Short entry: TK cross bearish (Tenkan < Kijun) AND price below cloud
            tk_cross_bearish = tenkan_sen_6h[i] < kijun_sen_6h[i]
            price_below_cloud = close[i] < cloud_bottom[i]
            
            if tk_cross_bullish and price_above_cloud:
                signals[i] = 0.25
                position = 1
            elif tk_cross_bearish and price_below_cloud:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TK cross bearish OR price drops below cloud
            if (tenkan_sen_6h[i] < kijun_sen_6h[i]) or (close[i] < cloud_top[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK cross bullish OR price rises above cloud
            if (tenkan_sen_6h[i] > kijun_sen_6h[i]) or (close[i] > cloud_bottom[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Ichimoku TK cross with cloud filter on 6h timeframe.
# Enters long when Tenkan-sen crosses above Kijun-sen AND price is above the cloud (bullish).
# Enters short when Tenkan-sen crosses below Kijun-sen AND price is below the cloud (bearish).
# Exits when TK cross reverses OR price re-enters the cloud.
# Uses Ichimoku from daily timeframe aligned to 6h to reduce noise and improve signal quality.
# Works in bull markets (trend-following with TK cross) and bear markets (counter-trend from cloud rejection).
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize churn.