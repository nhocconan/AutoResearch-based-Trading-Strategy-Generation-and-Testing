#!/usr/bin/env python3
name = "6h_1d_Ichimoku_Cloud_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components on daily chart
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2, plotted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2, plotted 26 periods ahead
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate cloud (Kumo) boundaries
    # Senkou Span A and B are already plotted 26 periods ahead in Ichimoku
    # So we need to align them as-is without additional shift
    upper_cloud = np.maximum(senkou_a_aligned, senkou_b_aligned)
    lower_cloud = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 4)  # Wait for Ichimoku and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or 
            np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TK cross bullish + price above cloud + volume
            tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
            price_above_cloud = close[i] > upper_cloud[i]
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            
            if tk_bullish and price_above_cloud and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish + price below cloud + volume
            elif (tenkan_aligned[i] < kijun_aligned[i] and 
                  close[i] < lower_cloud[i] and 
                  vol_condition):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TK cross bearish or price drops below cloud
            if (tenkan_aligned[i] < kijun_aligned[i] or 
                close[i] < lower_cloud[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross bullish or price rises above cloud
            if (tenkan_aligned[i] > kijun_aligned[i] or 
                close[i] > upper_cloud[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Ichimoku Cloud breakout with TK cross and volume confirmation
# - Ichimoku cloud acts as dynamic support/resistance in all market conditions
# - TK cross (Tenkan/Kijun crossover) signals momentum shift
# - Price above/below cloud confirms trend direction
# - Volume spike (2x average) validates breakout strength
# - Works in bull markets (buy cloud breaks in uptrend) and bear markets (sell cloud breaks in downtrend)
# - Cloud provides natural stop/resistance levels
# - Position size 0.25 targets ~20-40 trades/year, minimizing fee drag
# - Uses actual daily Ichimoku calculations (not approximated) for accuracy
# - Designed to work in BOTH bull and bear markets via cloud position and TK cross