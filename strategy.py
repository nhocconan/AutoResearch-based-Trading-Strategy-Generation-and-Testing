#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1wTrend_VolumeSpike
Hypothesis: Ichimoku TK cross (Tenkan/Kijun) with weekly trend filter (price > weekly Kumo top for longs, < weekly Kumo bottom for shorts), and volume spike (>2x median) to confirm institutional participation. Uses 6h timeframe to reduce trade frequency and avoid fee drag. Weekly trend ensures alignment with major market structure, while TK cross captures momentum shifts. Volume spike filters false breakouts. Designed for BTC/ETH: works in bull/bear by following weekly trend, avoiding counter-trend whipsaws. Targets 12-30 trades/year via strict confluence.
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
    
    # Load 1w data ONCE before loop for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    highest_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    highest_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 plotted 26 periods ahead
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 plotted 26 periods ahead
    period_senkou_b = 52
    highest_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (highest_senkou_b + lowest_senkou_b) / 2
    
    # Weekly trend filter: price relative to weekly Kumo (cloud)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Ichimoku cloud (using same periods scaled to weekly)
    # Weekly Tenkan: 9-period
    highest_tenkan_1w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    lowest_tenkan_1w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (highest_tenkan_1w + lowest_tenkan_1w) / 2
    
    # Weekly Kijun: 26-period
    highest_kijun_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    lowest_kijun_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_1w = (highest_kijun_1w + lowest_kijun_1w) / 2
    
    # Weekly Senkou Span A
    senkou_a_1w = (tenkan_1w + kijun_1w) / 2
    
    # Weekly Senkou Span B: 52-period
    highest_senkou_b_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    lowest_senkou_b_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = (highest_senkou_b_1w + lowest_senkou_b_1w) / 2
    
    # Weekly Kumo top (max of Senkou A/B) and bottom (min of Senkou A/B)
    kumo_top_1w = np.maximum(senkou_a_1w, senkou_b_1w)
    kumo_bottom_1w = np.minimum(senkou_a_1w, senkou_b_1w)
    
    # Align weekly Kumo to 6h (completed weekly candle only)
    kumo_top_aligned = align_htf_to_ltf(prices, df_1w, kumo_top_1w)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1w, kumo_bottom_1w)
    
    # Volume spike: volume > 2.0x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (2.0 * vol_median_20)
    
    # Fixed position size to control trade frequency and drawdown
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 52 for Senkou B, 20 for volume median
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        vol_spike = volume_spike[i]
        kumo_top = kumo_top_aligned[i]
        kumo_bottom = kumo_bottom_aligned[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # TK cross: Tenkan crosses above/below Kijun
            tk_cross_up = tenkan_val > kijun_val and tenkan[i-1] <= kijun[i-1]
            tk_cross_down = tenkan_val < kijun_val and tenkan[i-1] >= kijun[i-1]
            
            # Long: TK cross up, price above weekly Kumo (bullish trend), volume spike
            long_entry = tk_cross_up and (close_val > kumo_top) and vol_spike
            # Short: TK cross down, price below weekly Kumo (bearish trend), volume spike
            short_entry = tk_cross_down and (close_val < kumo_bottom) and vol_spike
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on TK cross down or price re-enters weekly Kumo (below cloud bottom)
            tk_cross_down = tenkan_val < kijun_val and tenkan[i-1] >= kijun[i-1]
            if tk_cross_down or close_val < kumo_bottom:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on TK cross up or price re-enters weekly Kumo (above cloud top)
            tk_cross_up = tenkan_val > kijun_val and tenkan[i-1] <= kijun[i-1]
            if tk_cross_up or close_val > kumo_top:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0