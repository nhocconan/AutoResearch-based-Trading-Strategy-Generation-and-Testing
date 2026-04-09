#!/usr/bin/env python3
# 6h_1d_1w_ichimoku_volume_v1
# Hypothesis: 6h strategy using Ichimoku cloud from 1d and 1w timeframes with volume confirmation.
# Enters long when price is above both daily and weekly clouds with TK cross bullish and volume spike.
# Enters short when price is below both daily and weekly clouds with TK cross bearish and volume spike.
# Uses volume spike (2x 20-period average) to filter false breaks. Discrete sizing (±0.25) to minimize fee churn.
# Ichimoku provides dynamic support/resistance and trend direction, working in both bull and bear markets via cloud twist.
# Target: 50-150 total trades over 4 years (12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_ichimoku_volume_v1"
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
    
    # Get 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Get 1w HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = ((period52_high + period52_low) / 2)
    
    # Calculate Ichimoku components for 1w
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high_1w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low_1w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (period9_high_1w + period9_low_1w) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_1w = (period26_high_1w + period26_low_1w) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a_1w = ((tenkan_1w + kijun_1w) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = ((period52_high_1w + period52_low_1w) / 2)
    
    # Align Ichimoku components to 6h timeframe (completed HTF candle only)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    # Volume spike detection (20-period volume average on 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or
            np.isnan(senkou_a_1w_aligned[i]) or np.isnan(senkou_b_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud_1d = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        lower_cloud_1d = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        upper_cloud_1w = np.maximum(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        lower_cloud_1w = np.minimum(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        
        # TK cross conditions
        tk_bullish_1d = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        tk_bearish_1d = tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        tk_bullish_1w = tenkan_1w_aligned[i] > kijun_1w_aligned[i]
        tk_bearish_1w = tenkan_1w_aligned[i] < kijun_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price falls below daily cloud OR TK cross turns bearish
            if (close[i] < lower_cloud_1d) or (not tk_bullish_1d):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above daily cloud OR TK cross turns bullish
            if (close[i] > upper_cloud_1d) or (not tk_bearish_1d):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above both clouds, TK bullish on both timeframes, volume spike
            if (close[i] > upper_cloud_1d and close[i] > upper_cloud_1w and
                tk_bullish_1d and tk_bullish_1w and vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price below both clouds, TK bearish on both timeframes, volume spike
            elif (close[i] < lower_cloud_1d and close[i] < lower_cloud_1w and
                  tk_bearish_1d and tk_bearish_1w and vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals