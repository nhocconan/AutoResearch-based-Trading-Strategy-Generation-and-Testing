#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_WeeklyTrend_v1
Hypothesis: 6h Ichimoku TK cross (Tenkan/Kijun) with Kumo twist confirmation and weekly trend filter.
In ranging markets, TK cross signals reversals when price is near Kumo edges. In trending markets,
TK cross with Kumo alignment captures continuation. Weekly trend filter (price vs weekly Kumo)
avoids counter-trend trades in strong weekly trends. Targets 50-150 trades over 4 years by requiring
TK cross, Kumo position filter, and weekly alignment - reducing false signals while capturing
both mean reversion and trend continuation across BTC/ETH market regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # --- 1d Ichimoku components (9, 26, 52 periods) ---
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (9-period)
    tenkan_9 = (pd.Series(high_1d).rolling(window=9, min_periods=9).mean() + 
                pd.Series(low_1d).rolling(window=9, min_periods=9).mean()) / 2
    tenkan_9 = tenkan_9.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_26 = (pd.Series(high_1d).rolling(window=26, min_periods=26).mean() + 
                pd.Series(low_1d).rolling(window=26, min_periods=26).mean()) / 2
    kijun_26 = kijun_26.values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan_9 + kijun_26) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_b = (pd.Series(high_1d).rolling(window=52, min_periods=52).mean() + 
                pd.Series(low_1d).rolling(window=52, min_periods=52).mean()) / 2
    senkou_b = senkou_b.values
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # For signal generation, we use current price vs Kumo (Senkou A/B)
    
    # Align 1d Ichimoku to 6h
    tenkan_9_aligned = align_htf_to_ltf(prices, df_1d, tenkan_9)
    kijun_26_aligned = align_htf_to_ltf(prices, df_1d, kijun_26)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # --- 1w Ichimoku for trend filter (weekly Kumo) ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Senkou Span A and B (using 9, 26, 52 but on weekly data)
    tenkan_9_1w = (pd.Series(high_1w).rolling(window=9, min_periods=9).mean() + 
                   pd.Series(low_1w).rolling(window=9, min_periods=9).mean()) / 2
    kijun_26_1w = (pd.Series(high_1w).rolling(window=26, min_periods=26).mean() + 
                   pd.Series(low_1w).rolling(window=26, min_periods=26).mean()) / 2
    senkou_a_1w = (tenkan_9_1w + kijun_26_1w) / 2
    senkou_b_1w = (pd.Series(high_1w).rolling(window=52, min_periods=52).mean() + 
                   pd.Series(low_1w).rolling(window=52, min_periods=52).mean()) / 2
    
    # Align weekly Ichimoku to 6h
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    # Volume confirmation: 6h volume > 1.5x 20-period average
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_9_aligned[i]) or 
            np.isnan(kijun_26_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i]) or
            np.isnan(senkou_a_1w_aligned[i]) or
            np.isnan(senkou_b_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Kumo twist detection: Senkou A and B cross (twist = trend change signal)
        # Kumo twist bullish: Senkou A crosses above Senkou B
        # Kumo twist bearish: Senkou A crosses below Senkou B
        senkou_a_prev = senkou_a_aligned[i-1] if i > 0 else senkou_a_aligned[i]
        senkou_b_prev = senkou_b_aligned[i-1] if i > 0 else senkou_b_aligned[i]
        
        kumo_twist_bullish = (senkou_a_prev <= senkou_b_prev) and (senkou_a_aligned[i] > senkou_b_aligned[i])
        kumo_twist_bearish = (senkou_a_prev >= senkou_b_prev) and (senkou_a_aligned[i] < senkou_b_aligned[i])
        
        # TK cross: Tenkan crosses Kijun
        tenkan_prev = tenkan_9_aligned[i-1] if i > 0 else tenkan_9_aligned[i]
        kijun_prev = kijun_26_aligned[i-1] if i > 0 else kijun_26_aligned[i]
        
        tk_cross_bullish = (tenkan_prev <= kijun_prev) and (tenkan_9_aligned[i] > kijun_26_aligned[i])
        tk_cross_bearish = (tenkan_prev >= kijun_prev) and (tenkan_9_aligned[i] < kijun_26_aligned[i])
        
        # Price relative to Kumo (cloud)
        kumo_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        kumo_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        price_above_kumo = close[i] > kumo_top
        price_below_kumo = close[i] < kumo_bottom
        price_in_kumo = (close[i] >= kumo_bottom) and (close[i] <= kumo_top)
        
        # Weekly trend filter: price vs weekly Kumo
        weekly_kumo_top = np.maximum(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        weekly_kumo_bottom = np.minimum(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        price_above_weekly_kumo = close[i] > weekly_kumo_top
        price_below_weekly_kumo = close[i] < weekly_kumo_bottom
        
        # Volume spike condition
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # LONG CONDITIONS:
        # 1. TK cross bullish
        # 2. Price above Kumo (bullish alignment) OR Kumo twist bullish (anticipation)
        # 3. Weekly trend alignment: price above weekly Kumo OR ranging (not strongly below)
        # 4. Volume confirmation
        if tk_cross_bullish and volume_spike:
            kumo_bullish = price_above_kumo or kumo_twist_bullish
            weekly_aligned = price_above_weekly_kumo or price_in_kumo  # Allow ranging in weekly Kumo
            if kumo_bullish and weekly_aligned:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
                continue
        
        # SHORT CONDITIONS:
        # 1. TK cross bearish
        # 2. Price below Kumo (bearish alignment) OR Kumo twist bearish
        # 3. Weekly trend alignment: price below weekly Kumo OR ranging (not strongly above)
        # 4. Volume confirmation
        if tk_cross_bearish and volume_spike:
            kumo_bearish = price_below_kumo or kumo_twist_bearish
            weekly_aligned = price_below_weekly_kumo or price_in_kumo  # Allow ranging in weekly Kumo
            if kumo_bearish and weekly_aligned:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
                continue
        
        # HOLD POSITION
        if position == 0:
            signals[i] = 0.0
        elif position == 1:
            signals[i] = 0.25
        else:
            signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_WeeklyTrend_v1"
timeframe = "6h"
leverage = 1.0