#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_Breakout_1dTrend_VolumeSpike
Hypothesis: Ichimoku TK cross (Tenkan/Kijun) with price above/below Kumo (cloud) from 6h, confirmed by 1d trend (price > 1d EMA50) and volume spike (top 30%). TK cross catches momentum, Kumo filter ensures trend alignment, volume spike confirms participation. Target: 12-30 trades/year on 6h.
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
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Ichimoku components on 6h (Tenkan = 9-period, Kijun = 26-period, Senkou A/B = 26/52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    highest_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (highest_9 + lowest_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    highest_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (highest_26 + lowest_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    period_senkou_b = 52
    highest_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((highest_52 + lowest_52) / 2)
    
    # Align Ichimoku components to current timeframe (no extra delay needed for TK cross)
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)  # same timeframe, no alignment needed
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b)
    
    # Kumo (cloud) top and bottom: max/min of Senkou A/B
    kumO_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    kumO_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # 1d trend filter: price > 1d EMA50 for long, price < 1d EMA50 for short
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume regime: volume > 70th percentile of 50-period lookback (high volume days only)
    vol_series = pd.Series(volume)
    vol_percentile_70 = vol_series.rolling(window=50, min_periods=50).quantile(0.70).values
    volume_regime = volume > vol_percentile_70
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (52 for Senkou B, 50 for EMA and volume percentile)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(kumO_top[i]) or np.isnan(kumO_bottom[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_percentile_70[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        kumO_top_val = kumO_top[i]
        kumO_bottom_val = kumO_bottom[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        vol_regime = volume_regime[i]
        size = fixed_size
        
        # TK cross: Tenkan crosses above/below Kijun
        tk_cross_up = (tenkan_val > kijun_val) and (tenkan_aligned[i-1] <= kijun_aligned[i-1])
        tk_cross_down = (tenkan_val < kijun_val) and (tenkan_aligned[i-1] >= kijun_aligned[i-1])
        
        # Price relative to Kumo: price above cloud (bullish) or below cloud (bearish)
        price_above_kumo = close_val > kumO_top_val
        price_below_kumo = close_val < kumO_bottom_val
        
        # Entry conditions: TK cross + price in correct Kumo + 1d trend alignment + volume regime
        long_entry = tk_cross_up and price_above_kumo and (close_val > ema_50_1d_val) and vol_regime
        short_entry = tk_cross_down and price_below_kumo and (close_val < ema_50_1d_val) and vol_regime
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when Tenkan crosses below Kijun (TK cross down) or price breaks below Kumo bottom
            if tk_cross_down or (close_val < kumO_bottom_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when Tenkan crosses above Kijun (TK cross up) or price breaks above Kumo top
            if tk_cross_up or (close_val > kumO_top_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0