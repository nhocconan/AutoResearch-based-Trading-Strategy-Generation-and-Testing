#!/usr/bin/env python3
"""
6h Weekly Trend + Daily Ichimoku TK Cross with Volume Filter
Hypothesis: Weekly Ichimoku cloud direction provides primary trend bias (works in both bull/bear via cloud color), 
daily Tenkan-Kijun cross provides precise entry timing, and volume > 1.5x average confirms institutional participation. 
Designed for 6h timeframe to avoid overtrading while capturing multi-timeframe alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_trend_daily_ichimoku_tk_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Ichimoku trend (primary filter)
    df_weekly = get_htf_data(prices, '1w')
    
    # Daily data for Ichimoku TK cross (entry signal)
    df_daily = get_htf_data(prices, '1d')
    
    # === WEEKLY ICHIMOKU CLOUD (trend filter) ===
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    wk_high9 = df_weekly['high'].rolling(window=9, min_periods=9).max()
    wk_low9 = df_weekly['low'].rolling(window=9, min_periods=9).min()
    wk_tenkan = (wk_high9 + wk_low9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    wk_high26 = df_weekly['high'].rolling(window=26, min_periods=26).max()
    wk_low26 = df_weekly['low'].rolling(window=26, min_periods=26).min()
    wk_kijun = (wk_high26 + wk_low26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    wk_senkou_a = ((wk_tenkan + wk_kijun) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    wk_high52 = df_weekly['high'].rolling(window=52, min_periods=52).max()
    wk_low52 = df_weekly['low'].rolling(window=52, min_periods=52).min()
    wk_senkou_b = ((wk_high52 + wk_low52) / 2).shift(26)
    
    # Cloud color: green if Senkou A > Senkou B (bullish), red otherwise (bearish)
    wk_cloud_green = wk_senkou_a > wk_senkou_b
    
    # Align weekly Ichimoku components to 6h
    wk_tenkan_6h = align_htf_to_ltf(prices, df_weekly, wk_tenkan.values)
    wk_kijun_6h = align_htf_to_ltf(prices, df_weekly, wk_kijun.values)
    wk_senkou_a_6h = align_htf_to_ltf(prices, df_weekly, wk_senkou_a.values)
    wk_senkou_b_6h = align_htf_to_ltf(prices, df_weekly, wk_senkou_b.values)
    wk_cloud_green_6h = align_htf_to_ltf(prices, df_weekly, wk_cloud_green.values)
    
    # === DAILY ICHIMOKU TK CROSS (entry signal) ===
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    dy_high9 = df_daily['high'].rolling(window=9, min_periods=9).max()
    dy_low9 = df_daily['low'].rolling(window=9, min_periods=9).min()
    dy_tenkan = (dy_high9 + dy_low9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    dy_high26 = df_daily['high'].rolling(window=26, min_periods=26).max()
    dy_low26 = df_daily['low'].rolling(window=26, min_periods=26).min()
    dy_kijun = (dy_high26 + dy_low26) / 2
    
    # TK Cross: 1 when Tenkan > Kijun (bullish cross), -1 when Tenkan < Kijun (bearish cross)
    dy_tk_cross = np.where(dy_tenkan > dy_kijun, 1, np.where(dy_tenkan < dy_kijun, -1, 0))
    
    # Align daily TK cross to 6h
    dy_tenkan_6h = align_htf_to_ltf(prices, df_daily, dy_tenkan.values)
    dy_kijun_6h = align_htf_to_ltf(prices, df_daily, dy_kijun.values)
    dy_tk_cross_6h = align_htf_to_ltf(prices, df_daily, dy_tk_cross)
    
    # Volume filter (>1.5x 24-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(wk_tenkan_6h[i]) or np.isnan(wk_kijun_6h[i]) or
            np.isnan(wk_senkou_a_6h[i]) or np.isnan(wk_senkou_b_6h[i]) or
            np.isnan(wk_cloud_green_6h[i]) or np.isnan(dy_tenkan_6h[i]) or
            np.isnan(dy_kijun_6h[i]) or np.isnan(dy_tk_cross_6h[i]) or
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from weekly cloud
        bullish_trend = wk_cloud_green_6h[i]  # Green cloud = bullish bias
        bearish_trend = not wk_cloud_green_6h[i]  # Red cloud = bearish bias
        
        if position == 1:  # Long position
            # Exit: TK cross turns bearish OR price exits cloud (contrarian signal)
            if dy_tk_cross_6h[i] == -1 or close[i] < wk_senkou_b_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TK cross turns bullish OR price enters cloud (contrarian signal)
            if dy_tk_cross_6h[i] == 1 or close[i] > wk_senkou_a_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: bullish weekly trend + bullish TK cross + volume
            if (bullish_trend and 
                dy_tk_cross_6h[i] == 1 and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: bearish weekly trend + bearish TK cross + volume
            elif (bearish_trend and 
                  dy_tk_cross_6h[i] == -1 and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals