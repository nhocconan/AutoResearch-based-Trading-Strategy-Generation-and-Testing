#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Trend_Signal
# Strategy: Trade Ichimoku cloud breakout with trend filter from 1d timeframe
# Long when price breaks above Kumo (cloud) and Tenkan > Kijun
# Short when price breaks below Kumo (cloud) and Tenkan < Kijun
# Exit when price re-enters Kumo
# Uses Ichimoku from 6h chart and trend filter from 1d to avoid counter-trend trades
# Designed for 6h timeframe with selective entries to minimize trade frequency
# Ichimoku components calculated with proper periods: Tenkan=9, Kijun=26, SenkouA/B=26, Chikou=26

name = "6h_Ichimoku_Cloud_Trend_Signal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Shift Senkou spans forward by 26 periods
    senkou_a_shifted = np.roll(senkou_a, period_kijun)
    senkou_b_shifted = np.roll(senkou_b, period_kijun)
    # Fill first 26 values with NaN
    senkou_a_shifted[:period_kijun] = np.nan
    senkou_b_shifted[:period_kijun] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = period_kijun + period_senkou_b  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_shifted[i]) or np.isnan(senkou_b_shifted[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou A and B shifted)
        upper_cloud = max(senkou_a_shifted[i], senkou_b_shifted[i])
        lower_cloud = min(senkou_a_shifted[i], senkou_b_shifted[i])
        
        if position == 0:
            # Enter long: price above cloud and bullish TK cross (Tenkan > Kijun) with uptrend filter
            if close[i] > upper_cloud and tenkan[i] > kijun[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price below cloud and bearish TK cross (Tenkan < Kijun) with downtrend filter
            elif close[i] < lower_cloud and tenkan[i] < kijun[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price re-enters cloud (below upper cloud)
            if close[i] < upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price re-enters cloud (above lower cloud)
            if close[i] > lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals