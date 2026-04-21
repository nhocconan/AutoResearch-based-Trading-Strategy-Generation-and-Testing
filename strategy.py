#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_v1
Hypothesis: Ichimoku Tenkan-Kijun cross with cloud filter on 6h, aligned with 1d EMA50 trend. 
Tenkan(9)/Kijun(26) cross provides timely momentum signals, while cloud (Senkou Span A/B) 
acts as dynamic support/resistance filter. 1d EMA50 ensures alignment with daily trend. 
Designed for low trade frequency (~15-30/year) to minimize fee drag. Works in bull/bear 
markets by only taking trades in direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d trend filter: 50-period EMA ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Ichimoku components on 6h ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    highest_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    highest_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    highest_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((highest_senkou_b + lowest_senkou_b) / 2)
    
    # Current cloud boundaries (Senkou Span A/B from 26 periods ago)
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Cloud top/bottom
    cloud_top = np.maximum(senkou_a_lagged, senkou_b_lagged)
    cloud_bottom = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        trend_1d = ema_50_1d_aligned[i]
        
        # Bullish TK cross: Tenkan crosses above Kijun
        tk_bullish_cross = (tenkan_val > kijun_val) and (tenkan[i-1] <= kijun[i-1])
        # Bearish TK cross: Tenkan crosses below Kijun
        tk_bearish_cross = (tenkan_val < kijun_val) and (tenkan[i-1] >= kijun[i-1])
        
        if position == 0:
            # Long: Bullish TK cross + price above cloud + price above 1d EMA50
            if tk_bullish_cross and price_close > cloud_top_val and price_close > trend_1d:
                signals[i] = 0.25
                position = 1
            # Short: Bearish TK cross + price below cloud + price below 1d EMA50
            elif tk_bearish_cross and price_close < cloud_bottom_val and price_close < trend_1d:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            if position == 1:
                # Exit long: price closes below cloud bottom OR Tenkan crosses below Kijun
                if price_close < cloud_bottom_val or (tenkan_val < kijun_val and tenkan[i-1] >= kijun[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price closes above cloud top OR Tenkan crosses above Kijun
                if price_close > cloud_top_val or (tenkan_val > kijun_val and tenkan[i-1] <= kijun[i-1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_v1"
timeframe = "6h"
leverage = 1.0