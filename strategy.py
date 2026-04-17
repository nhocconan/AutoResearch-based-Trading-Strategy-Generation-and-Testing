#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_Trend
Strategy: 6h Ichimoku Kumo twist with volume confirmation and weekly trend filter.
Long: Tenkan > Kijun + price above Kumo + Kumo future twist bullish + weekly uptrend
Short: Tenkan < Kijun + price below Kumo + Kumo future twist bearish + weekly downtrend
Exit: Tenkan/Kijun cross reversal or Kumo break
Position size: 0.25
Designed to catch strong trends while avoiding chop using Ichimoku's multi-line confirmation.
Timeframe: 6h
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
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    highest_tenkan = np.maximum.accumulate(high)
    lowest_tenkan = np.minimum.accumulate(low)
    # For rolling window, use pandas for simplicity
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    tenkan = (high_series.rolling(window=tenkan_period, center=False).max() + 
              low_series.rolling(window=tenkan_period, center=False).min()) / 2
    tenkan = tenkan.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun = (high_series.rolling(window=kijun_period, center=False).max() + 
             low_series.rolling(window=kijun_period, center=False).min()) / 2
    kijun = kijun.values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_b = (high_series.rolling(window=senkou_span_b_period, center=False).max() + 
                low_series.rolling(window=senkou_span_b_period, center=False).min()) / 2
    senkou_b = senkou_b.values
    
    # Future Kumo twist (bullish/bearish): compare current Senkou A/B with displaced Senkou A/B
    # Kumo twist bullish: Senkou A > Senkou B (future)
    # Kumo twist bearish: Senkou A < Senkou B (future)
    # We need to compare current Senkou with Senkou from 'displacement' periods ago
    senkou_a_lagged = np.roll(senkou_a, displacement)
    senkou_b_lagged = np.roll(senkou_b, displacement)
    # First 'displacement' values are invalid due to roll
    senkou_a_lagged[:displacement] = np.nan
    senkou_b_lagged[:displacement] = np.nan
    
    kumo_twist_bullish = senkou_a > senkou_b_lagged  # Future Kumo bullish twist
    kumo_twist_bearish = senkou_a < senkou_b_lagged  # Future Kumo bearish twist
    
    # Get weekly trend (close > open = uptrend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    trend_1w = (df_1w['close'] > df_1w['open']).astype(float).values  # 1 for up, 0 for down
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from sufficient warmup
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period, displacement, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(trend_1w_aligned[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Current volume
        volume_current = volume[i]
        volume_filter = volume_current > (1.5 * volume_ma20[i])
        
        # Ichimoku conditions
        price_above_kumo = close[i] > max(senkou_a[i], senkou_b[i])
        price_below_kumo = close[i] < min(senkou_a[i], senkou_b[i])
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        
        # Entry signals
        if position == 0:
            # Long: Tenkan > Kijun + price above Kumo + bullish Kumo twist + weekly uptrend + volume
            if (tenkan_above_kijun and price_above_kumo and kumo_twist_bullish[i] and 
                trend_1w_aligned[i] > 0.5 and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan < Kijun + price below Kumo + bearish Kumo twist + weekly downtrend + volume
            elif (tenkan_below_kijun and price_below_kumo and kumo_twist_bearish[i] and 
                  trend_1w_aligned[i] < 0.5 and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Tenkan/Kijun cross down OR price breaks below Kumo
            if not tenkan_above_kijun or not price_above_kumo:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Tenkan/Kijun cross up OR price breaks above Kumo
            if not tenkan_below_kijun or not price_below_kumo:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_Trend"
timeframe = "6h"
leverage = 1.0