#!/usr/bin/env python3
# Hypothesis: 6h Ichimoku Cloud with daily trend filter - Long when price above cloud and TK cross bullish with daily close > weekly VWAP, short when price below cloud and TK cross bearish with daily close < weekly VWAP
# Uses Ichimoku for trend/momentum and weekly VWAP for institutional bias. Works in bull/bear by following institutional flow.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "6h_Ichimoku_Cloud_1dWeeklyVWAP"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components (9, 26, 52 periods)
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # Not used for signals to avoid look-ahead
    
    # Get 1d data for weekly VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly VWAP from daily data
    # Approximate weekly VWAP using 5-day period
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_num = (typical_price * df_1d['volume']).rolling(window=5, min_periods=5).sum()
    vwap_den = df_1d['volume'].rolling(window=5, min_periods=5).sum()
    weekly_vwap = vwap_num / vwap_den
    
    # Align Ichimoku and weekly VWAP to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, prices, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, prices, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, prices, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, prices, senkou_span_b.values)
    weekly_vwap_aligned = align_htf_to_ltf(prices, df_1d, weekly_vwap.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, senkou_span_b_period)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(weekly_vwap_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # TK cross signals
        tk_cross_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_cross_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        if position == 0:
            # Enter long: price above cloud, TK cross bullish, and price above weekly VWAP
            if (close[i] > cloud_top and 
                tk_cross_bullish and
                close[i] > weekly_vwap_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below cloud, TK cross bearish, and price below weekly VWAP
            elif (close[i] < cloud_bottom and 
                  tk_cross_bearish and
                  close[i] < weekly_vwap_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below cloud or TK cross turns bearish
            if close[i] < cloud_bottom or not tk_cross_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above cloud or TK cross turns bullish
            if close[i] > cloud_top or tk_cross_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals