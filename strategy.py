# The hypothesis: 12h price action above/below weekly Ichimoku Cloud (Tenkan/Kijun) with volume confirmation and weekly trend filter (price above/below weekly Kijun-Sen).
# Uses weekly Ichimoku for trend direction and 12h for entry timing, designed to work in both bull and bear markets by capturing strong momentum moves with proper trend alignment.
# Targets 15-25 trades/year with position size 0.25.

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
    
    # Get weekly data for Ichimoku
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 12h timeframe (wait for weekly bar close)
    tenkan_12h = align_htf_to_ltf(prices, df_1w, tenkan)
    kijun_12h = align_htf_to_ltf(prices, df_1w, kijun)
    senkou_a_12h = align_htf_to_ltf(prices, df_1w, senkou_a)
    senkou_b_12h = align_htf_to_ltf(prices, df_1w, senkou_b)
    
    # Calculate volume average (20-period) for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # need Ichimoku data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_12h[i]) or np.isnan(kijun_12h[i]) or 
            np.isnan(senkou_a_12h[i]) or np.isnan(senkou_b_12h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_12h[i], senkou_b_12h[i])
        cloud_bottom = min(senkou_a_12h[i], senkou_b_12h[i])
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price above cloud AND above Kijun with volume confirmation
            if close[i] > cloud_top and close[i] > kijun_12h[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price below cloud AND below Kijun with volume confirmation
            elif close[i] < cloud_bottom and close[i] < kijun_12h[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below Kijun or cloud bottom
            if close[i] < kijun_12h[i] or close[i] < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Kijun or cloud top
            if close[i] > kijun_12h[i] or close[i] > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Ichimoku_Cloud_Breakout_With_Volume_Filter"
timeframe = "12h"
leverage = 1.0