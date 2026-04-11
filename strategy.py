#!/usr/bin/env python3
"""
6h_1w_Ichimoku_Trend_Follow
Hypothesis: Uses Ichimoku cloud from weekly timeframe for trend direction and 6h price action for entry.
Trades only in direction of weekly Kumo (cloud) to avoid counter-trend whipsaws.
Weekly Senkou Span A/B forms dynamic support/resistance. Price above/below cloud determines trend.
Tenkan/Kijun cross on 6s for entry timing. Works in bull/bear by following weekly trend.
Target: 20-50 trades/year per symbol with high win rate during strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_Ichimoku_Trend_Follow"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load weekly data ONCE before loop for Ichimoku
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    high_9 = pd.Series(high_1w).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_9 = pd.Series(low_1w).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    high_26 = pd.Series(high_1w).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_26 = pd.Series(low_1w).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    high_52 = pd.Series(high_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_52 = pd.Series(low_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods back
    # Not used for signals but calculated for completeness
    
    # Align Ichimoku components to 6h timeframe (wait for weekly bar to close)
    tenkan_6h = align_htf_to_ltf(prices, df_1w, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1w, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1w, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1w, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Senkou B period
        # Skip if any required data is invalid
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine if price is above or below weekly cloud
        # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
        cloud_top = np.maximum(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = np.minimum(senkou_a_6h[i], senkou_b_6h[i])
        
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK cross signals (Tenkan crosses Kijun)
        tk_cross_up = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
        tk_cross_down = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
        
        # Entry conditions: trade in direction of weekly trend (cloud)
        long_entry = price_above_cloud and tk_cross_up
        short_entry = price_below_cloud and tk_cross_down
        
        # Exit conditions: reverse TK cross or price enters cloud
        long_exit = tk_cross_down or (close[i] <= cloud_top and close[i] >= cloud_bottom)
        short_exit = tk_cross_up or (close[i] <= cloud_top and close[i] >= cloud_bottom)
        
        # Priority: entry > exit > hold
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
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals