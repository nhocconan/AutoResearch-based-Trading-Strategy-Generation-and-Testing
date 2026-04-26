#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1wTrend_HTFVolSpike
Hypothesis: Ichimoku TK cross with 1w trend filter and HTF volume spike for 6h timeframe.
Long when: TK cross bullish + price above cloud + 1w EMA50 uptrend + 1d volume > 2.0 * 20-period avg.
Short when: TK cross bearish + price below cloud + 1w EMA50 downtrend + 1d volume > 2.0 * 20-period avg.
Exit when: TK cross reverses or price crosses Kijun-sen.
Uses discrete 0.25 position size. Targets 12-25 trades/year to avoid fee drag.
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
    
    # Calculate Ichimoku components on 6h data
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
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind (not used for entry)
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d volume spike: current volume > 2.0 * 20-period average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (2.0 * vol_avg_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 52 for Senkou B, 50 for 1w EMA, 20 for volume avg
    start_idx = max(52, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or
            np.isnan(senkou_b[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        if position == 0:
            # Flat - look for TK cross with trend and volume confirmation
            # Bullish TK cross: Tenkan crosses above Kijun
            tk_bullish = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
            # Bearish TK cross: Tenkan crosses below Kijun
            tk_bearish = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
            
            # Long: bullish TK + price above cloud + 1w EMA50 uptrend + volume spike
            long_entry = tk_bullish and (close_val > cloud_top) and \
                       (ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]) and \
                       volume_spike_1d_aligned[i]
            # Short: bearish TK + price below cloud + 1w EMA50 downtrend + volume spike
            short_entry = tk_bearish and (close_val < cloud_bottom) and \
                        (ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]) and \
                        volume_spike_1d_aligned[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when TK cross turns bearish or price crosses below Kijun
            tk_bearish = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
            price_below_kijun = close_val < kijun[i]
            if tk_bearish or price_below_kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when TK cross turns bullish or price crosses above Kijun
            tk_bullish = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
            price_above_kijun = close_val > kijun[i]
            if tk_bullish or price_above_kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_1wTrend_HTFVolSpike"
timeframe = "6h"
leverage = 1.0