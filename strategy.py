#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1wTrend_HTFVolSpike_v1
Hypothesis: Use 6h timeframe with Ichimoku TK cross (Tenkan/Kijun) from 1d HTF, confirmed by 1w EMA50 trend and 1w volume spike. Targets 12-30 trades/year. Works in both bull and bear markets by using 1w trend filter to avoid counter-trend signals and volume spike to ensure participation. TK cross provides timely entries while 1w filter ensures higher timeframe alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Calculate Ichimoku components from 1d HTF
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need 26*2 for Senkou Span B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Align TK cross to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1w volume spike: current volume > 2.0 * 20-period average
    vol_1w = df_1w['volume'].values
    vol_avg_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    volume_spike_1w = vol_1w > (2.0 * vol_avg_1w)
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 26 for Kijun, 50 for 1w EMA, 20 for volume avg
    start_idx = max(26, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = prices['close'].iloc[i]
        size = 0.25  # 25% position size to manage risk and trade frequency
        
        if position == 0:
            # Flat - look for TK cross with 1w trend and volume confirmation
            # Long: Tenkan crosses above Kijun + 1w EMA50 uptrend + volume spike
            tk_cross_up = (tenkan_aligned[i] > kijun_aligned[i]) and (tenkan_aligned[i-1] <= kijun_aligned[i-1])
            ema_uptrend = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            long_entry = tk_cross_up and ema_uptrend and volume_spike_1w_aligned[i]
            
            # Short: Tenkan crosses below Kijun + 1w EMA50 downtrend + volume spike
            tk_cross_down = (tenkan_aligned[i] < kijun_aligned[i]) and (tenkan_aligned[i-1] >= kijun_aligned[i-1])
            ema_downtrend = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
            short_entry = tk_cross_down and ema_downtrend and volume_spike_1w_aligned[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when Tenkan crosses below Kijun (TK cross reversal)
            tk_cross_down = (tenkan_aligned[i] < kijun_aligned[i]) and (tenkan_aligned[i-1] >= kijun_aligned[i-1])
            if tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when Tenkan crosses above Kijun (TK cross reversal)
            tk_cross_up = (tenkan_aligned[i] > kijun_aligned[i]) and (tenkan_aligned[i-1] <= kijun_aligned[i-1])
            if tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_1wTrend_HTFVolSpike_v1"
timeframe = "6h"
leverage = 1.0