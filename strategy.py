#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_Breakout_1dTrend_VolumeFilter
Hypothesis: On 6h timeframe, enter long when price breaks above Kumo (cloud) AND Tenkan/Kijun twist bullish (Tenkan > Kijun) AND 1d trend up (close > EMA50) AND volume > 1.5x 20-period average. Enter short on opposite conditions. Uses Ichimoku cloud as dynamic support/resistance with twist filter for momentum confirmation and 1d trend alignment. Designed for 6h timeframe to capture medium-term swings with lower trade frequency (12-37/year) and strong edge in both bull and bear markets via trend-filtered breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou = 52
    max_high_senkou = pd.Series(high).rolling(window=period_senkou, min_periods=period_senkou).max().values
    min_low_senkou = pd.Series(low).rolling(window=period_senkou, min_periods=period_senkou).min().values
    senkou_b = (max_high_senkou + min_low_senkou) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)  # same timeframe
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b)
    
    # Kumo (cloud) boundaries: Senkou Span A and B
    upper_kumo = np.maximum(senkou_a_aligned, senkou_b_aligned)
    lower_kumo = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Tenkan (9), Kijun (26), Senkou B (52), volume MA (20)
    start_idx = max(9, 26, 52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(upper_kumo[i]) or np.isnan(lower_kumo[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Ichimoku conditions
        price_above_kumo = close[i] > upper_kumo[i]
        price_below_kumo = close[i] < lower_kumo[i]
        tenkan_above_kijun = tenkan_aligned[i] > kijun_aligned[i]  # bullish twist
        tenkan_below_kijun = tenkan_aligned[i] < kijun_aligned[i]  # bearish twist
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_50_1d_aligned[i]
        trend_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price above Kumo + bullish twist + 1d uptrend + volume spike
            long_signal = price_above_kumo and tenkan_above_kijun and trend_uptrend and volume_spike[i]
            
            # Short: price below Kumo + bearish twist + 1d downtrend + volume spike
            short_signal = price_below_kumo and tenkan_below_kijun and trend_downtrend and volume_spike[i]
            
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
            # Exit: price breaks below Kumo OR twist turns bearish OR trend change to downtrend
            if price_below_kumo or not tenkan_above_kijun or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above Kumo OR twist turns bullish OR trend change to uptrend
            if price_above_kumo or not tenkan_below_kijun or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_Breakout_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0