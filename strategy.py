#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_1wTrend
Hypothesis: Ichimoku Tenkan-Kijun cross with cloud filter on 6h timeframe, aligned with 1week trend (price above/below weekly Kumo) captures medium-term momentum while avoiding counter-trend trades. Works in bull markets (trend-following TK crosses above cloud) and bear markets (trend-following TK crosses below cloud). Target: 12-30 trades/year to minimize fee drag on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Ichimoku components (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku components: Tenkan-sen (9-period), Kijun-sen (26-period), Senkou Span A/B (52-period)
    # Tenkan-sen: (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen: (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A: (Tenkan + Kijun) / 2, plotted 26 periods ahead
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B: (52-period high + 52-period low) / 2, plotted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe (completed 1d bar)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # 1w data for weekly trend filter (Kumo - cloud)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly Ichimoku cloud for trend filter
    whigh = df_1w['high'].values
    wlow = df_1w['low'].values
    wclose = df_1w['close'].values
    
    # Weekly Tenkan-sen and Kijun-sen
    wk_period9_high = pd.Series(whigh).rolling(window=9, min_periods=9).max().values
    wk_period9_low = pd.Series(wlow).rolling(window=9, min_periods=9).min().values
    w_tenkan = (wk_period9_high + wk_period9_low) / 2
    
    wk_period26_high = pd.Series(whigh).rolling(window=26, min_periods=26).max().values
    wk_period26_low = pd.Series(wlow).rolling(window=26, min_periods=26).min().values
    w_kijun = (wk_period26_high + wk_period26_low) / 2
    
    # Weekly Senkou Span A and B
    w_senkou_a = (w_tenkan + w_kijun) / 2
    wk_period52_high = pd.Series(whigh).rolling(window=52, min_periods=52).max().values
    wk_period52_low = pd.Series(wlow).rolling(window=52, min_periods=52).min().values
    w_senkou_b = (wk_period52_high + wk_period52_low) / 2
    
    # Align weekly cloud to 6h timeframe (completed 1w bar)
    w_senkou_a_aligned = align_htf_to_ltf(prices, df_1w, w_senkou_a)
    w_senkou_b_aligned = align_htf_to_ltf(prices, df_1w, w_senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku calculations
    start_idx = max(52, 26, 9) + 26  # +26 for forward displacement of Senkou Span
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(w_senkou_a_aligned[i]) or np.isnan(w_senkou_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Cloud boundaries (Senkou Span A/B)
        upper_cloud = max(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Weekly cloud boundaries for trend filter
        w_upper_cloud = max(w_senkou_a_aligned[i], w_senkou_b_aligned[i])
        w_lower_cloud = min(w_senkou_a_aligned[i], w_senkou_b_aligned[i])
        
        if position == 0:
            # Look for entry signals - TK cross with cloud alignment and weekly trend filter
            tk_cross_bull = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            tk_cross_bear = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
            
            # Price above/below cloud
            price_above_cloud = curr_close > upper_cloud
            price_below_cloud = curr_close < lower_cloud
            
            # Weekly trend filter: price relative to weekly cloud
            weekly_uptrend = curr_close > w_upper_cloud
            weekly_downtrend = curr_close < w_lower_cloud
            
            # Long entry: bullish TK cross above cloud + weekly uptrend
            if tk_cross_bull and price_above_cloud and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish TK cross below cloud + weekly downtrend
            elif tk_cross_bear and price_below_cloud and weekly_downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when TK cross turns bearish OR price falls below cloud
            tk_cross_bear = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
            price_below_cloud = curr_close < lower_cloud
            
            if tk_cross_bear or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when TK cross turns bullish OR price rises above cloud
            tk_cross_bull = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            price_above_cloud = curr_close > upper_cloud
            
            if tk_cross_bull or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1wTrend"
timeframe = "6h"
leverage = 1.0