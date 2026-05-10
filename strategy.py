#!/usr/bin/env python3
# 4h_Ichimoku_Tenkan_Kijun_Cross_With_Volume
# Hypothesis: Long when Tenkan-sen crosses above Kijun-sen with volume > 1.5x average and price > Kumo (cloud).
# Short when Tenkan-sen crosses below Kijun-sen with volume > 1.5x average and price < Kumo.
# Exit when Tenkan-sen crosses back over Kijun-sen or price enters Kumo.
# Uses Ichimoku Cloud for trend and support/resistance, effective in both bull and bear markets.
# Designed for 20-50 trades/year to avoid fee drag.

name = "4h_Ichimoku_Tenkan_Kijun_Cross_With_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku Cloud components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    
    period_tenkan = 9
    period_kijun = 26
    period_senkou_b = 52
    
    # Calculate highest high and lowest low for each period
    highest_high_9 = np.full(n, np.nan)
    lowest_low_9 = np.full(n, np.nan)
    highest_high_26 = np.full(n, np.nan)
    lowest_low_26 = np.full(n, np.nan)
    highest_high_52 = np.full(n, np.nan)
    lowest_low_52 = np.full(n, np.nan)
    
    for i in range(n):
        if i >= period_tenkan - 1:
            highest_high_9[i] = np.max(high[i - period_tenkan + 1:i + 1])
            lowest_low_9[i] = np.min(low[i - period_tenkan + 1:i + 1])
        if i >= period_kijun - 1:
            highest_high_26[i] = np.max(high[i - period_kijun + 1:i + 1])
            lowest_low_26[i] = np.min(low[i - period_kijun + 1:i + 1])
        if i >= period_senkou_b - 1:
            highest_high_52[i] = np.max(high[i - period_senkou_b + 1:i + 1])
            lowest_low_52[i] = np.min(low[i - period_senkou_b + 1:i + 1])
    
    # Tenkan-sen and Kijun-sen
    tenkan = np.full(n, np.nan)
    kijun = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(highest_high_9[i]) and not np.isnan(lowest_low_9[i]):
            tenkan[i] = (highest_high_9[i] + lowest_low_9[i]) / 2
        if not np.isnan(highest_high_26[i]) and not np.isnan(lowest_low_26[i]):
            kijun[i] = (highest_high_26[i] + lowest_low_26[i]) / 2
    
    # Senkou Span A and B (shifted forward)
    senkou_a = np.full(n, np.nan)
    senkou_b = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            senkou_a[i] = (tenkan[i] + kijun[i]) / 2
        if not np.isnan(highest_high_52[i]) and not np.isnan(lowest_low_52[i]):
            senkou_b[i] = (highest_high_52[i] + lowest_low_52[i]) / 2
    
    # Shift Senkou Spans forward by 26 periods
    senkou_a_shifted = np.full(n, np.nan)
    senkou_b_shifted = np.full(n, np.nan)
    for i in range(n):
        if i + period_kijun < n:
            senkou_a_shifted[i + period_kijun] = senkou_a[i]
            senkou_b_shifted[i + period_kijun] = senkou_b[i]
    
    # Kumo (cloud) boundaries: Senkou Span A and B
    # Kumo top is the higher of Senkou A and B
    # Kumo bottom is the lower of Senkou A and B
    kumo_top = np.full(n, np.nan)
    kumo_bottom = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(senkou_a_shifted[i]) and not np.isnan(senkou_b_shifted[i]):
            kumo_top[i] = max(senkou_a_shifted[i], senkou_b_shifted[i])
            kumo_bottom[i] = min(senkou_a_shifted[i], senkou_b_shifted[i])
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = period_kijun + period_senkou_b  # Ensure sufficient warmup for Ichimoku
    
    for i in range(start_idx, n):
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Tenkan/Kijun cross
            if i > 0 and not np.isnan(tenkan[i-1]) and not np.isnan(kijun[i-1]):
                # Bullish cross: Tenkan crosses above Kijun
                if tenkan[i-1] <= kijun[i-1] and tenkan[i] > kijun[i]:
                    # Long: price above Kumo and volume confirmation
                    if close[i] > kumo_top[i] and volume[i] > 1.5 * vol_ma[i]:
                        signals[i] = 0.25
                        position = 1
                # Bearish cross: Tenkan crosses below Kijun
                elif tenkan[i-1] >= kijun[i-1] and tenkan[i] < kijun[i]:
                    # Short: price below Kumo and volume confirmation
                    if close[i] < kumo_bottom[i] and volume[i] > 1.5 * vol_ma[i]:
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:
            # Exit: Tenkan crosses back below Kijun or price enters Kumo
            if (i > 0 and not np.isnan(tenkan[i-1]) and not np.isnan(kijun[i-1]) and 
                tenkan[i-1] > kijun[i-1] and tenkan[i] <= kijun[i]) or \
               close[i] < kumo_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Tenkan crosses back above Kijun or price enters Kumo
            if (i > 0 and not np.isnan(tenkan[i-1]) and not np.isnan(kijun[i-1]) and 
                tenkan[i-1] < kijun[i-1] and tenkan[i] >= kijun[i]) or \
               close[i] > kumo_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals