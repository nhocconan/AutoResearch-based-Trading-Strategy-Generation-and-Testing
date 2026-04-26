#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: 6h Ichimoku cloud breakout with 1d trend filter (Tenkan > Kijun) and volume spike confirmation.
- Long when price breaks above Ichimoku cloud (Senkou Span A/B) AND 1d Tenkan > Kijun AND volume > 2.0 * volume_ma(20)
- Short when price breaks below Ichimoku cloud AND 1d Tenkan < Kijun AND volume > 2.0 * volume_ma(20)
- Uses Ichimoku from 6h chart for structure-based breakouts
- 1d Tenkan/Kijun filter ensures trading with higher timeframe momentum to avoid counter-trend whipsaws
- Volume spike (2.0x) confirms institutional participation and reduces false breakouts
- Exit when price re-enters the cloud or 1d momentum shifts
- Designed for moderate frequency (target 12-37 trades/year on 6h) to minimize fee drag
- Novelty: Ichimoku cloud breakouts on 6h with 1d trend filter - different from saturated Camarilla/Donchian variants
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Tenkan-sen (9-period) and Kijun-sen (26-period) for trend filter
    # Tenkan-sen = (Highest High + Lowest Low) / 2 over past 9 periods
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    tenkan_1d = (high_1d.rolling(window=9, min_periods=9).max() + 
                 low_1d.rolling(window=9, min_periods=9).min()) / 2.0
    # Kijun-sen = (Highest High + Lowest Low) / 2 over past 26 periods
    kijun_1d = (high_1d.rolling(window=26, min_periods=26).max() + 
                low_1d.rolling(window=26, min_periods=26).min()) / 2.0
    tenkan_1d_vals = tenkan_1d.values
    kijun_1d_vals = kijun_1d.values
    # Align to 6h timeframe (wait for completed 1d candle)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d_vals)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d_vals)
    # 1d momentum: 1 = bullish (Tenkan > Kijun), -1 = bearish (Tenkan < Kijun), 0 = neutral/invalid
    momentum_1d = np.where(tenkan_1d_aligned > kijun_1d_aligned, 1,
                           np.where(tenkan_1d_aligned < kijun_1d_aligned, -1, 0))
    
    # Calculate Ichimoku components on 6h chart (primary timeframe)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_6h = pd.Series(high)
    low_6h = pd.Series(low)
    tenkan_6h = (high_6h.rolling(window=9, min_periods=9).max() + 
                 low_6h.rolling(window=9, min_periods=9).min()) / 2.0
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_6h = (high_6h.rolling(window=26, min_periods=26).max() + 
                low_6h.rolling(window=26, min_periods=26).min()) / 2.0
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_6h + kijun_6h) / 2.0).shift(2)  # shifted 2 periods ahead
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_span_b = ((high_6h.rolling(window=52, min_periods=52).max() + 
                      low_6h.rolling(window=52, min_periods=52).min()) / 2.0).shift(2)
    
    tenkan_6h_vals = tenkan_6h.values
    kijun_6h_vals = kijun_6h.values
    senkou_span_a_vals = senkou_span_a.values
    senkou_span_b_vals = senkou_span_b.values
    
    # Calculate volume filter: volume > 2.0 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B, 26 for Kijun, 20 for volume MA)
    start_idx = max(52, 26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(senkou_span_a_vals[i]) or np.isnan(senkou_span_b_vals[i]) or
            np.isnan(momentum_1d[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = max(senkou_span_a_vals[i], senkou_span_b_vals[i])
        lower_cloud = min(senkou_span_a_vals[i], senkou_span_b_vals[i])
        
        # Ichimoku cloud breakout conditions with 1d momentum and volume spike filter
        if position == 0:
            # Long: Price breaks above cloud AND 1d bullish momentum AND volume spike
            if close[i] > upper_cloud and momentum_1d[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below cloud AND 1d bearish momentum AND volume spike
            elif close[i] < lower_cloud and momentum_1d[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below cloud OR 1d momentum turns bearish
            if close[i] < lower_cloud or momentum_1d[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above cloud OR 1d momentum turns bullish
            if close[i] > upper_cloud or momentum_1d[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0