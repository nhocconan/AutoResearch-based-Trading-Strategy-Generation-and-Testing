#!/usr/bin/env python3
"""
6h_1d_1w_Ichimoku_Cloud_Breakout_v1
Hypothesis: Use Ichimoku Cloud from daily timeframe for trend and support/resistance,
with weekly timeframe for higher-timeframe trend confirmation. Enter long when price breaks above cloud
with bullish TK cross and weekly trend up, short when breaks below cloud with bearish TK cross and weekly trend down.
Uses volume confirmation to avoid false breakouts. Targets 15-25 trades/year to minimize fee drag.
Ichimoku works well in both trending and ranging markets by providing dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_Ichimoku_Cloud_Breakout_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: tenkan, kijun, senkouA, senkouB, chikou"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = pd.Series(close).shift(26).values
    
    return tenkan, kijun, senkou_a, senkou_b, chikou

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === DAILY ICHIMOKU CLOUD ===
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(d_high, d_low, d_close)
    
    # Cloud top and bottom (Senkou Span A and B)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Align to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    cloud_top_6h = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_6h = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    
    # TK Cross signals
    tk_cross_bullish = tenkan_6h > kijun_6h
    tk_cross_bearish = tenkan_6h < kijun_6h
    
    # Price above/below cloud
    price_above_cloud = close > cloud_top_6h
    price_below_cloud = close < cloud_bottom_6h
    
    # === WEEKLY EMA25 TREND FILTER ===
    w_close = df_1w['close'].values
    ema25 = pd.Series(w_close).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema25_6h = align_htf_to_ltf(prices, df_1w, ema25)
    weekly_uptrend = close > ema25_6h
    weekly_downtrend = close < ema25_6h
    
    # === VOLUME FILTER (24-period average) ===
    vol_ma = np.full(n, np.nan)
    if n >= 24:
        vol_sum = np.sum(volume[:24])
        vol_ma[23] = vol_sum / 24
        for i in range(24, n):
            vol_sum = vol_sum - volume[i-24] + volume[i]
            vol_ma[i] = vol_sum / 24
    
    volume_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any data invalid
        if (np.isnan(cloud_top_6h[i]) or np.isnan(cloud_bottom_6h[i]) or 
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(ema25_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions
        long_entry = (price_above_cloud[i] and 
                     tk_cross_bullish[i] and 
                     weekly_uptrend[i] and 
                     volume_confirm[i])
        
        short_entry = (price_below_cloud[i] and 
                      tk_cross_bearish[i] and 
                      weekly_downtrend[i] and 
                      volume_confirm[i])
        
        # Exit conditions: reverse signal or price returns to Kijun
        long_exit = not price_above_cloud[i] or close[i] < kijun_6h[i]
        short_exit = not price_below_cloud[i] or close[i] > kijun_6h[i]
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals