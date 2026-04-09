#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1w/1d trend filter and volume confirmation
# Uses Ichimoku TK cross as entry signal with cloud filter from higher timeframe
# In bull markets: long when price > cloud + TK cross up + volume
# In bear markets: short when price < cloud + TK cross down + volume
# Weekly trend filter ensures we only trade with the major trend
# Daily volume confirmation reduces false signals
# Target: 50-150 total trades over 4 years = 12-37/year
# Discrete position sizing 0.25 to limit fee drag

name = "6h_1w_1d_ichimoku_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    # Not used for signals as it requires future data
    
    # Calculate weekly trend filter using Ichimoku
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Tenkan-sen and Kijun-sen
    wk_period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    wk_period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    wk_tenkan = (wk_period9_high + wk_period9_low) / 2
    
    wk_period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    wk_period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    wk_kijun = (wk_period26_high + wk_period26_low) / 2
    
    # Weekly cloud
    wk_senkou_a = (wk_tenkan + wk_kijun) / 2
    wk_period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    wk_period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    wk_senkou_b = (wk_period52_high + wk_period52_low) / 2
    
    # Daily volume confirmation
    volume_1d = df_1d['volume'].values
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan_sen)  # same timeframe
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_span_b)
    
    wk_tenkan_aligned = align_htf_to_ltf(prices, df_1w, wk_tenkan)
    wk_kijun_aligned = align_htf_to_ltf(prices, df_1w, wk_kijun)
    wk_senkou_a_aligned = align_htf_to_ltf(prices, df_1w, wk_senkou_a)
    wk_senkou_b_aligned = align_htf_to_ltf(prices, df_1w, wk_senkou_b)
    
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(wk_tenkan_aligned[i]) or np.isnan(wk_kijun_aligned[i]) or
            np.isnan(wk_senkou_a_aligned[i]) or np.isnan(wk_senkou_b_aligned[i]) or
            np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine Ichimoku cloud boundaries (future cloud values)
        upper_cloud = max(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Weekly trend: bullish if Tenkan > Kijun AND price above weekly cloud
        wk_upper_cloud = max(wk_senkou_a_aligned[i], wk_senkou_b_aligned[i])
        wk_lower_cloud = min(wk_senkou_a_aligned[i], wk_senkou_b_aligned[i])
        weekly_bullish = (wk_tenkan_aligned[i] > wk_kijun_aligned[i] and 
                         close[i] > wk_upper_cloud)
        weekly_bearish = (wk_tenkan_aligned[i] < wk_kijun_aligned[i] and 
                         close[i] < wk_lower_cloud)
        
        # Volume confirmation: current volume > 1.5x daily average
        volume_confirmed = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        # TK cross signals
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
        
        if position == 1:  # Long position
            # Exit conditions: TK cross down OR price below cloud OR weekly trend turns bearish
            if (tk_cross_down or close[i] < lower_cloud or not weekly_bullish):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: TK cross up OR price above cloud OR weekly trend turns bullish
            if (tk_cross_up or close[i] > upper_cloud or not weekly_bearish):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: TK cross in direction of weekly trend with volume confirmation
            if weekly_bullish and tk_cross_up and volume_confirmed and close[i] > upper_cloud:
                position = 1
                signals[i] = 0.25
            elif weekly_bearish and tk_cross_down and volume_confirmed and close[i] < lower_cloud:
                position = -1
                signals[i] = -0.25
    
    return signals