#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_WeeklyTrend_Direction
Hypothesis: 6h Ichimoku cloud breakout with 1w trend filter and volume confirmation.
In uptrend (price > 1w Kumo top): long when price breaks above Kumo top with volume spike.
In downtrend (price < 1w Kumo bottom): short when price breaks below Kumo bottom with volume spike.
Exit when price re-enters the cloud or trend reverses. Designed to capture strong trends
while avoiding choppy markets. Uses Ichimoku's built-in trend/filter properties.
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
    
    # Get 6h data for Ichimoku calculations (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:  # need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high_6h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low_6h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    high_kijun = pd.Series(high_6h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low_6h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou_b + low_senkou_b) / 2)
    
    # Align Ichimoku components to original timeframe (with proper delay)
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 26:  # need at least 26 periods for Ichimoku
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Ichimoku for trend direction
    high_tenkan_1w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_tenkan_1w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (high_tenkan_1w + low_tenkan_1w) / 2
    
    high_kijun_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_kijun_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_1w = (high_kijun_1w + low_kijun_1w) / 2
    
    senkou_a_1w = ((tenkan_1w + kijun_1w) / 2)
    high_senkou_b_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_senkou_b_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = ((high_senkou_b_1w + low_senkou_b_1w) / 2)
    
    # Kumo top/bottom for 1w trend
    kumo_top_1w = np.maximum(senkou_a_1w, senkou_b_1w)
    kumo_bottom_1w = np.minimum(senkou_a_1w, senkou_b_1w)
    
    kumo_top_1w_aligned = align_htf_to_ltf(prices, df_1w, kumo_top_1w)
    kumo_bottom_1w_aligned = align_htf_to_ltf(prices, df_1w, kumo_bottom_1w)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(kumo_top_1w_aligned[i]) or np.isnan(kumo_bottom_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current Kumo boundaries
        kumo_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        kumo_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # 1w trend direction
        weekly_uptrend = close[i] > kumo_top_1w_aligned[i]
        weekly_downtrend = close[i] < kumo_bottom_1w_aligned[i]
        
        if position == 0:
            # Look for breakout signals with volume confirmation
            if weekly_uptrend:
                # Long: price breaks above Kumo top with volume spike
                long_signal = (close[i] > kumo_top) and vol_spike[i]
                # Short: price breaks below Kumo bottom only with extreme volume (counter-trend)
                short_signal = (close[i] < kumo_bottom) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            elif weekly_downtrend:
                # Short: price breaks below Kumo bottom with volume spike
                short_signal = (close[i] < kumo_bottom) and vol_spike[i]
                # Long: price breaks above Kumo top only with extreme volume (counter-trend)
                long_signal = (close[i] > kumo_top) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            else:
                # Weekly sideways/choppy: no new positions
                long_signal = False
                short_signal = False
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price re-enters Kumo or weekly trend turns down
            exit_signal = (close[i] < kumo_top) or (close[i] > kumo_bottom) or weekly_downtrend
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price re-enters Kumo or weekly trend turns up
            exit_signal = (close[i] > kumo_bottom) or (close[i] < kumo_top) or weekly_uptrend
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_WeeklyTrend_Direction"
timeframe = "6h"
leverage = 1.0