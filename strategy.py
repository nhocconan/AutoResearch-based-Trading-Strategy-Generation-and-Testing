#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1wTrend_Filter
Hypothesis: On 6h timeframe, Ichimoku TK cross (Tenkan/Kijun) signals aligned with 1week cloud color (bullish/bearish) provide high-probability entries.
Uses 1week trend filter to avoid counter-trend whipsaws in ranging/bear markets. Discrete position sizing (0.0, ±0.25) to limit fee churn.
Designed for 12-25 trades/year per symbol. Works in bull/bear via 1week trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need ~1 year of weekly data for Ichimoku
        return np.zeros(n)
    
    # === 6h Ichimoku Components ===
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high_6h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low_6h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    high_kijun = pd.Series(high_6h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low_6h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou_b + low_senkou_b) / 2.0)
    
    # Chikou Span (Lagging Span): close shifted -26 periods (not used for signals)
    
    # === 1week Cloud Color (Trend Filter) ===
    # Cloud color: bullish when Senkou A > Senkou B, bearish when Senkou A < Senkou B
    # We need 1week Ichimoku to determine cloud color
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1week Tenkan-sen (9-period)
    high_tenkan_1w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_tenkan_1w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (high_tenkan_1w + low_tenkan_1w) / 2.0
    
    # 1week Kijun-sen (26-period)
    high_kijun_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_kijun_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_1w = (high_kijun_1w + low_kijun_1w) / 2.0
    
    # 1week Senkou Span A
    senkou_a_1w = ((tenkan_1w + kijun_1w) / 2.0)
    
    # 1week Senkou Span B (52-period)
    high_senkou_b_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_senkou_b_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = ((high_senkou_b_1w + low_senkou_b_1w) / 2.0)
    
    # Cloud color: 1 = bullish (A > B), -1 = bearish (A < B), 0 = undefined
    cloud_color_1w = np.where(senkou_a_1w > senkou_b_1w, 1, np.where(senkou_a_1w < senkou_b_1w, -1, 0))
    
    # Align HTF indicators to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan)  # Not actually used for signal but for completeness
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_b)
    cloud_color_aligned = align_htf_to_ltf(prices, df_1w, cloud_color_1w, additional_delay_bars=0)  # Cloud color known at bar close
    
    # Align 6h Ichimoku (calculated on 6h data) to 6h prices (no delay needed as it's based on current/completed 6h bar)
    # Note: Since we're calculating Ichimoku on 6h data, we don't need HTF alignment for the 6h components
    # But we do need to align the 1week cloud color
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_6h - low_6h)
    tr2 = pd.Series(np.abs(high_6h - np.roll(close_6h, 1)))
    tr3 = pd.Series(np.abs(low_6h - np.roll(close_6h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if indicators not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(cloud_color_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        
        if position == 0:
            # TK Cross signals
            tk_cross_bullish = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
            tk_cross_bearish = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
            
            # Cloud filter: only trade in direction of 1week cloud
            cloud_bullish = cloud_color_aligned[i] > 0
            cloud_bearish = cloud_color_aligned[i] < 0
            
            # Entry logic: TK cross aligned with 1week cloud color
            if tk_cross_bullish and cloud_bullish:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif tk_cross_bearish and cloud_bearish:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions
            # Exit on bearish TK cross
            tk_cross_bearish = tenchan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
            # Exit on price cloud break (price below Senkou B)
            price_cloud_break = price < senkou_b[i]
            
            if tk_cross_bearish or price_cloud_break:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions
            # Exit on bullish TK cross
            tk_cross_bullish = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
            # Exit on price cloud break (price above Senkou A)
            price_cloud_break = price > senkou_a[i]
            
            if tk_cross_bullish or price_cloud_break:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0