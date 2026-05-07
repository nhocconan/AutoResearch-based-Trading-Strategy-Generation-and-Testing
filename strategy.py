#!/usr/bin/env python3
name = "6h_1d_Ichimoku_TK_Cross_Cloud_Filter_v2"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Volume confirmation: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 4)  # Wait for Ichimoku and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price above/below cloud
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        
        # TK cross
        tk_cross_bull = tenkan_aligned[i] > kijun_aligned[i]
        tk_cross_bear = tenkan_aligned[i] < kijun_aligned[i]
        
        if position == 0:
            # Long: price above cloud + TK bullish cross + volume confirmation
            vol_condition = volume[i] > vol_ma_4[i] * 1.5
            if price_above_cloud and tk_cross_bull and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK bearish cross + volume confirmation
            elif price_below_cloud and tk_cross_bear and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price drops below cloud or TK bearish cross
            if close[i] < cloud_top[i] or (tenkan_aligned[i] < kijun_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises above cloud or TK bullish cross
            if close[i] > cloud_bottom[i] or (tenkan_aligned[i] > kijun_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Ichimoku TK cross with cloud filter and volume confirmation
# - Ichimoku cloud acts as dynamic support/resistance
# - TK cross above/below cloud signals momentum shift
# - Volume confirmation (1.5x average) filters false signals
# - Works in both bull (buy above cloud in uptrend) and bear (sell below cloud in downtrend)
# - Cloud provides natural stop-loss equivalent
# - Position size 0.25 targets ~15-35 trades/year, avoiding fee drag
# - Uses daily Ichimoku for higher timeframe structure
# - Improved over prior attempt by adding volume filter and refining exit conditions