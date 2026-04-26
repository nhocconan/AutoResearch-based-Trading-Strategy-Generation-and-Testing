#!/usr/bin/env python3
"""
6h_IchiCloud_Trend_1dEMA50_Filter
Hypothesis: Use Ichimoku cloud (from 1d) as trend filter + TK cross on 6h for entry, with volume confirmation. Works in bull (cloud support) and bear (cloud resistance) by only trading in direction of higher timeframe trend. Target 12-30 trades/year.
"""

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
    open_time = prices['open_time'].values
    
    # Session filter: UTC 8-20 for institutional activity
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Ichimoku cloud and EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_52 + min_low_52) / 2)
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    chikou = close_1d
    
    # Calculate cloud (Senkou Span A/B) - note: already shifted 26 periods ahead in calculation
    # For current price, we need Senkou A/B that were calculated 26 periods ago
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_lagged, senkou_b_lagged)
    cloud_bottom = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    # 1d EMA50 for additional trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1d indicators to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    chikou_aligned = align_htf_to_ltf(prices, df_1d, chikou)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # TK cross on 6h: Tenkan/Kijun cross
    tk_cross_above = tenkan_aligned > kijun_aligned
    tk_cross_below = tenkan_aligned < kijun_aligned
    
    # Price relative to cloud
    price_above_cloud = close > cloud_top_aligned
    price_below_cloud = close < cloud_bottom_aligned
    price_in_cloud = ~(price_above_cloud | price_below_cloud)
    
    # Chikou confirmation: Chikou above/below price 26 periods ago
    chikou_confirm_long = chikou_aligned > np.roll(close, 26)
    chikou_confirm_short = chikou_aligned < np.roll(close, 26)
    # Handle first 26 values
    chikou_confirm_long[:26] = False
    chikou_confirm_short[:26] = False
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku calculations (52) and volume MA (20)
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top_aligned[i]) or 
            np.isnan(cloud_bottom_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter: price and EMA50 on same side of cloud
        bullish_trend = (close[i] > ema_50_1d_aligned[i]) and price_above_cloud[i]
        bearish_trend = (close[i] < ema_50_1d_aligned[i]) and price_below_cloud[i]
        
        if position == 0:
            # Long: TK cross bullish + bullish trend + chikou long + volume
            long_signal = (tk_cross_above[i] and 
                          bullish_trend and 
                          chikou_confirm_long[i] and 
                          volume_confirmed[i])
            # Short: TK cross bearish + bearish trend + chikou short + volume
            short_signal = (tk_cross_below[i] and 
                           bearish_trend and 
                           chikou_confirm_short[i] and 
                           volume_confirmed[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TK cross bearish OR price drops below cloud bottom OR trend reversal
            if (tk_cross_below[i] or 
                close[i] < cloud_bottom_aligned[i] or
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross bullish OR price rises above cloud top OR trend reversal
            if (tk_cross_above[i] or 
                close[i] > cloud_top_aligned[i] or
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_IchiCloud_Trend_1dEMA50_Filter"
timeframe = "6h"
leverage = 1.0