#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with weekly trend filter and volume confirmation.
In bull markets (weekly price > weekly Kumo top): long when price crosses above Tenkan-sen and price > Kijun-sen.
In bear markets (weekly price < weekly Kumo bottom): short when price crosses below Tenkan-sen and price < Kijun-sen.
Volume must be above 20-period average to confirm momentum.
Ichimoku provides dynamic support/resistance (Kumo) and momentum signals (TK cross).
Weekly trend filter ensures alignment with higher timeframe direction.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_weekly_trend_volume_v1"
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
    
    # === WEEKLY TREND FILTER (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    weekly_tenkan_high = pd.Series(weekly_high).rolling(window=9, min_periods=9).max().values
    weekly_tenkan_low = pd.Series(weekly_low).rolling(window=9, min_periods=9).min().values
    weekly_tenkan = (weekly_tenkan_high + weekly_tenkan_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    weekly_kijun_high = pd.Series(weekly_high).rolling(window=26, min_periods=26).max().values
    weekly_kijun_low = pd.Series(weekly_low).rolling(window=26, min_periods=26).min().values
    weekly_kijun = (weekly_kijun_high + weekly_kijun_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    weekly_senkou_a = (weekly_tenkan + weekly_kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    weekly_senkou_b_high = pd.Series(weekly_high).rolling(window=52, min_periods=52).max().values
    weekly_senkou_b_low = pd.Series(weekly_low).rolling(window=52, min_periods=52).min().values
    weekly_senkou_b = (weekly_senkou_b_high + weekly_senkou_b_low) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods back
    # Not used in signals but needed for Kumo calculation
    
    # Align weekly Ichimoku components to 6h timeframe
    weekly_tenkan_aligned = align_htf_to_ltf(prices, df_1w, weekly_tenkan)
    weekly_kijun_aligned = align_htf_to_ltf(prices, df_1w, weekly_kijun)
    weekly_senkou_a_aligned = align_htf_to_ltf(prices, df_1w, weekly_senkou_a)
    weekly_senkou_b_aligned = align_htf_to_ltf(prices, df_1w, weekly_senkou_b)
    
    # Kumo (Cloud) boundaries: Senkou Span A and B
    weekly_kumo_top = np.maximum(weekly_senkou_a_aligned, weekly_senkou_b_aligned)
    weekly_kumo_bottom = np.minimum(weekly_senkou_a_aligned, weekly_senkou_b_aligned)
    
    # === 6H ICHIMOKU COMPONENTS (LTF) ===
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    tenkan_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    tenkan_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    kijun_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    kijun_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (kijun_high + kijun_low) / 2
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after warmup for weekly Senkou B (52 periods)
        if np.isnan(weekly_kumo_top[i]) or np.isnan(weekly_kumo_bottom[i]) or \
           np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend relative to Kumo
        price_above_weekly_kumo = close[i] > weekly_kumo_top[i]
        price_below_weekly_kumo = close[i] < weekly_kumo_bottom[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Tenkan-sen OR weekly trend turns bearish (price below Kumo)
            if close[i] < tenkan[i] or price_below_weekly_kumo:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Tenkan-sen OR weekly trend turns bullish (price above Kumo)
            if close[i] > tenkan[i] or price_above_weekly_kumo:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # TK Cross: Tenkan-sen crossing Kijun-sen
            tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
            tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
            
            # Entry logic based on weekly trend (Kumo position)
            if price_above_weekly_kumo:
                # Bullish weekly trend: look for long on TK cross up
                if tk_cross_up:
                    position = 1
                    signals[i] = 0.25
            elif price_below_weekly_kumo:
                # Bearish weekly trend: look for short on TK cross down
                if tk_cross_down:
                    position = -1
                    signals[i] = -0.25
    
    return signals