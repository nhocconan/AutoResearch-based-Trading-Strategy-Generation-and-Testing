#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeSpike
Hypothesis: On 6h timeframe, enter long when price breaks above Kumo cloud AND Tenkan > Kijun AND 1d trend is up (close > EMA50) AND volume > 1.8x 20-period average volume. Enter short when price breaks below Kumo cloud AND Tenkan < Kijun AND 1d trend is down (close < EMA50) AND volume spike. Exit on Kumo cross reverse. Ichimoku provides dynamic support/resistance with trend/momentum confirmation, reducing false breakouts in ranging markets. Targets 12-30 trades/year on BTC/ETH/SOL with controlled fee drag.
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
    
    # Current Kumo cloud boundaries (Senkou Span A/B from 26 periods ago)
    # Since Senkou spans are plotted 26 periods ahead, current cloud is from 26 periods ago
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Kumo top and bottom
    kumo_top = np.maximum(senkou_a_lagged, senkou_b_lagged)
    kumo_bottom = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    # Volume confirmation: volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Ichimoku warmup (52) + Senkou shift (26) + EMA warmup (50) + volume MA (20)
    start_idx = max(52 + 26, 50, 20)  # 78
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or 
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Ichimoku conditions
        price_above_kumo = close[i] > kumo_top[i]
        price_below_kumo = close[i] < kumo_bottom[i]
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_50_1d_aligned[i]
        trend_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price above Kumo + Tenkan > Kijun + volume spike + 1d uptrend
            long_signal = price_above_kumo and tenkan_above_kijun and volume_spike[i] and trend_uptrend
            
            # Short: price below Kumo + Tenkan < Kijun + volume spike + 1d downtrend
            short_signal = price_below_kumo and tenkan_below_kijun and volume_spike[i] and trend_downtrend
            
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
            # Exit: price falls below Kumo bottom OR Tenkan < Kijun (trend weakness)
            if close[i] < kumo_bottom[i] or tenkan[i] < kijun[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above Kumo top OR Tenkan > Kijun (trend strength)
            if close[i] > kumo_top[i] or tenkan[i] > kijun[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0