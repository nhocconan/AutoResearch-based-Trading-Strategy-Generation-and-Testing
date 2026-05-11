#!/usr/bin/env python3
# 6h_1d_Ichimoku_Cloud_Trend_Breakout
# Hypothesis: Uses 1-day Ichimoku cloud (from daily timeframe) as trend filter and 6h price breakouts above/below cloud for entries.
# In bull markets, price above cloud + breakout above Senkou Span A captures momentum.
# In bear markets, price below cloud + breakout below Senkou Span B captures downside moves.
# Uses volume confirmation to avoid false breakouts. Target: 15-30 trades/year to minimize fee drag.

name = "6h_1d_Ichimoku_Cloud_Trend_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1-day data for Ichimoku cloud calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1-day Ichimoku Cloud Components ---
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe (no additional delay needed as Ichimoku uses current bar data)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # --- Volume confirmation (1.5x 24-period average on 6h) ---
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Ichimoku calculations (52 periods for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        # Determine cloud top and bottom (Senkou Span A and B)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: price breaks above cloud top with volume surge and bullish alignment (Tenkan > Kijun)
            if (close[i] > cloud_top and 
                volume_surge and 
                tenkan_sen_aligned[i] > kijun_sen_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud bottom with volume surge and bearish alignment (Tenkan < Kijun)
            elif (close[i] < cloud_bottom and 
                  volume_surge and 
                  tenkan_sen_aligned[i] < kijun_sen_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below cloud bottom OR Tenkan crosses below Kijun
                if (close[i] < cloud_bottom or 
                    tenkan_sen_aligned[i] < kijun_sen_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above cloud top OR Tenkan crosses above Kijun
                if (close[i] > cloud_top or 
                    tenkan_sen_aligned[i] > kijun_sen_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals