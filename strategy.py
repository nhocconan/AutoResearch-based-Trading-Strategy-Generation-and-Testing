#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with TK cross and 1d trend filter
# Uses Kumo (cloud) from 1d timeframe as trend filter: price above cloud = bullish bias, below = bearish bias
# TK cross (Tenkan/Kijun) on 6h for entry timing with confirmation from cloud color
# Designed to capture trends in both bull and bear markets with controlled trade frequency
# Ichimoku components: Tenkan (9-period), Kijun (26-period), Senkou Span A/B (26/52-period)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((high_52 + low_52) / 2)
    
    # Get 1d data for trend filter (cloud from daily timeframe)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Ichimoku cloud components
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max()
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min()
    tenkan_1d = (high_9_1d + low_9_1d) / 2
    
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max()
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min()
    kijun_1d = (high_26_1d + low_26_1d) / 2
    
    # Senkou Span A and B for 1d cloud
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max()
    low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min()
    senkou_b_1d = ((high_52_1d + low_52_1d) / 2)
    
    # Align 1d cloud to 6h timeframe (no additional delay needed for Ichimoku)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    signals = np.zeros(n)
    
    for i in range(52, n):  # Start after warmup for Ichimoku calculations
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i])):
            continue
        
        # Determine cloud boundaries and color from 1d data
        cloud_top = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_green = senkou_a_1d_aligned[i] > senkou_b_1d_aligned[i]  # bullish cloud
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK cross signals
        tk_cross_up = tenkan[i] > kijun[i] and (i == 0 or tenkan[i-1] <= kijun[i-1])
        tk_cross_down = tenkan[i] < kijun[i] and (i == 0 or tenkan[i-1] >= kijun[i-1])
        
        # Long: Price above cloud + bullish cloud + TK cross up
        if price_above_cloud and cloud_green and tk_cross_up:
            signals[i] = 0.25
        
        # Short: Price below cloud + bearish cloud + TK cross down
        elif price_below_cloud and not cloud_green and tk_cross_down:
            signals[i] = -0.25
        
        # Exit: TK cross in opposite direction or price enters cloud
        elif i > 0:
            exit_condition = (
                (signals[i-1] == 0.25 and (tk_cross_down or not price_above_cloud)) or
                (signals[i-1] == -0.25 and (tk_cross_up or not price_below_cloud)) or
                (signals[i-1] == 0.25 and close[i] <= cloud_top) or
                (signals[i-1] == -0.25 and close[i] >= cloud_bottom)
            )
            if exit_condition:
                signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_IchimokuCloud_TKCross_1dFilter"
timeframe = "6h"
leverage = 1.0