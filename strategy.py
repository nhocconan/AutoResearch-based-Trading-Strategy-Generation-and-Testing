# [68799] 6h_1d_Ichimoku_TK_Cross_Cloud_Filter_v1
# Hypothesis: Ichimoku TK cross with cloud filter on daily timeframe provides reliable trend signals.
# In bull markets: price above cloud + TK cross bullish = long signal.
# In bear markets: price below cloud + TK cross bearish = short signal.
# The cloud acts as dynamic support/resistance, reducing whipsaws.
# Timeframe: 6h, HTF: 1d for Ichimoku calculations.
# Target: 20-50 trades/year, low frequency to avoid fee drag.
# Position size: 0.25 (discrete levels to minimize churn).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for Ichimoku calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 6h timeframe (with proper look-ahead avoidance)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if NaN in critical values
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]  # Use daily close for signal generation (aligned to 6h via Ichimoku)
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = min(senkou_a_6h[i], senkou_b_6h[i])
        
        if position == 0:
            # Long: price above cloud + TK cross bullish (Tenkan > Kijun)
            if (price > cloud_top and tenkan_6h[i] > kijun_6h[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK cross bearish (Tenkan < Kijun)
            elif (price < cloud_bottom and tenkan_6h[i] < kijun_6h[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price drops below cloud OR TK cross turns bearish
            if price < cloud_bottom or tenkan_6h[i] < kijun_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above cloud OR TK cross turns bullish
            if price > cloud_top or tenkan_6h[i] > kijun_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Ichimoku_TK_Cross_Cloud_Filter_v1"
timeframe = "6h"
leverage = 1.0