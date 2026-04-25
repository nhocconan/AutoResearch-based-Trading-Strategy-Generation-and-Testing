#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_v4
Hypothesis: Trade 6h Ichimoku TK Cross with 1d cloud filter and volume confirmation.
- Ichimoku components: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (26, 52 displaced)
- TK Cross: Tenkan crosses above/below Kijun as momentum signal
- Cloud filter: price above/below 1d Ichimoku cloud for trend alignment
- Volume confirmation: require volume > 1.8x 20-period average
- Only take longs in bullish cloud (price > cloud), shorts in bearish cloud (price < cloud)
- Exit on TK cross reversal or cloud color change
- Position size: 0.25. Target: 50-150 total trades over 4 years = 12-37/year.
- Works in both bull and bear: Ichimoku adapts to volatility, cloud acts as dynamic support/resistance.
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
    
    # Get 1d data for HTF Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate Ichimoku on 6h (for TK Cross)
    period_tenkan = 9
    period_kijun = 26
    period_senkou = 52
    
    # Tenkan-sen: (HH+LL)/2 over 9 periods
    high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_9 + low_9) / 2.0
    
    # Kijun-sen: (HH+LL)/2 over 26 periods
    high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_26 + low_26) / 2.0
    
    # Senkou Span A: (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    
    # Senkou Span B: (HH+LL)/2 over 52 periods shifted 26 periods ahead
    high_52 = pd.Series(high).rolling(window=period_senkou, min_periods=period_senkou).max().values
    low_52 = pd.Series(low).rolling(window=period_senkou, min_periods=period_senkou).min().values
    senkou_b = ((high_52 + low_52) / 2.0)
    
    # Calculate 1d Ichimoku cloud for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Tenkan (9), Kijun (26)
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2.0
    
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26_1d + low_26_1d) / 2.0
    
    # 1d Senkou Span A/B
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2.0)
    high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = ((high_52_1d + low_52_1d) / 2.0)
    
    # Align 1d Ichimoku components to 6h
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Cloud boundaries: max/min of Senkou Span A/B
    cloud_top = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    # TK Cross signals
    tk_cross_above = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_cross_below = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for indicators
    start_idx = max(period_kijun, period_senkou, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine cloud relationship
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        
        if position == 0:
            # Entry: TK cross in direction of cloud alignment with volume confirmation
            long_setup = tk_cross_above[i] and price_above_cloud and volume_spike[i]
            short_setup = tk_cross_below[i] and price_below_cloud and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit on TK cross bearish or price drops below cloud
            exit_signal = tk_cross_below[i] or (close[i] < cloud_top[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit on TK cross bullish or price rises above cloud
            exit_signal = tk_cross_above[i] or (close[i] > cloud_bottom[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_v4"
timeframe = "6h"
leverage = 1.0