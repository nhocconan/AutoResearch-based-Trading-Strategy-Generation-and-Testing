#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Ichimoku_Cloud_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Ichimoku components and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Daily trend filter: price above/below 50 EMA
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    trend_up = close_1d > ema50_1d
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for Ichimoku (52 periods)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(trend_up_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud top and bottom
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long entry: TK cross bullish + price above cloud + daily uptrend
            tk_cross_bull = tenkan_aligned[i] > kijun_aligned[i]
            price_above_cloud = close[i] > cloud_top
            
            if tk_cross_bull and price_above_cloud and trend_up_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            
            # Short entry: TK cross bearish + price below cloud + daily downtrend
            tk_cross_bear = tenkan_aligned[i] < kijun_aligned[i]
            price_below_cloud = close[i] < cloud_bottom
            
            if tk_cross_bear and price_below_cloud and trend_up_aligned[i] < 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below cloud OR TK cross bearish
            if close[i] < cloud_bottom or tenkan_aligned[i] < kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above cloud OR TK cross bullish
            if close[i] > cloud_top or tenkan_aligned[i] > kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Ichimoku Cloud system on daily timeframe with TK cross signals and trend filter on 6H timeframe.
# The Ichimoku cloud provides dynamic support/resistance, TK cross signals momentum shifts.
# Works in bull markets (trend following with cloud support) and bear markets (counter-trend at cloud edges).
# Daily EMA50 filter ensures alignment with longer-term trend, reducing counter-trend trades.
# Target: 15-25 trades/year to minimize fee decay while capturing significant momentum shifts.
# Uses 6H timeframe for execution with daily Ichimoku filters to avoid overtrading.