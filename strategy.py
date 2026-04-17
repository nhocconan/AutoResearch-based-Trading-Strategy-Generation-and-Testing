# 6h_Ichimoku_Kumo_Twist_Trend_Filter_v1
# Ichimoku system with Kumo twist detection and trend filter on 6h timeframe
# Uses TK cross + cloud color + Kumo twist (Senkou Span A/B cross) for high-probability entries
# Designed to work in both bull and bear markets by filtering with trend direction
# Target: 50-150 total trades over 4 years (12-37/year)

#!/usr/bin/env python3
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
    
    # === 1d Ichimoku Components ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    highest_high_tenkan = np.full_like(high_1d, np.nan)
    lowest_low_tenkan = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= period_tenkan - 1:
            highest_high_tenkan[i] = np.max(high_1d[i-(period_tenkan-1):i+1])
            lowest_low_tenkan[i] = np.min(low_1d[i-(period_tenkan-1):i+1])
        elif i > 0:
            highest_high_tenkan[i] = np.max(high_1d[0:i+1])
            lowest_low_tenkan[i] = np.min(low_1d[0:i+1])
        else:
            highest_high_tenkan[i] = high_1d[0]
            lowest_low_tenkan[i] = low_1d[0]
    
    tenkan_sen = np.full_like(high_1d, np.nan)
    for i in range(len(high_1d)):
        if not (np.isnan(highest_high_tenkan[i]) or np.isnan(lowest_low_tenkan[i])):
            tenkan_sen[i] = (highest_high_tenkan[i] + lowest_low_tenkan[i]) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    highest_high_kijun = np.full_like(high_1d, np.nan)
    lowest_low_kijun = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= period_kijun - 1:
            highest_high_kijun[i] = np.max(high_1d[i-(period_kijun-1):i+1])
            lowest_low_kijun[i] = np.min(low_1d[i-(period_kijun-1):i+1])
        elif i > 0:
            highest_high_kijun[i] = np.max(high_1d[0:i+1])
            lowest_low_kijun[i] = np.min(low_1d[0:i+1])
        else:
            highest_high_kijun[i] = high_1d[0]
            lowest_low_kijun[i] = low_1d[0]
    
    kijun_sen = np.full_like(high_1d, np.nan)
    for i in range(len(high_1d)):
        if not (np.isnan(highest_high_kijun[i]) or np.isnan(lowest_low_kijun[i])):
            kijun_sen[i] = (highest_high_kijun[i] + lowest_low_kijun[i]) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = np.full_like(high_1d, np.nan)
    for i in range(len(high_1d)):
        if not (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i])):
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    highest_high_senkou_b = np.full_like(high_1d, np.nan)
    lowest_low_senkou_b = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= period_senkou_b - 1:
            highest_high_senkou_b[i] = np.max(high_1d[i-(period_senkou_b-1):i+1])
            lowest_low_senkou_b[i] = np.min(low_1d[i-(period_senkou_b-1):i+1])
        elif i > 0:
            highest_high_senkou_b[i] = np.max(high_1d[0:i+1])
            lowest_low_senkou_b[i] = np.min(low_1d[0:i+1])
        else:
            highest_high_senkou_b[i] = high_1d[0]
            lowest_low_senkou_b[i] = low_1d[0]
    
    senkou_span_b = np.full_like(high_1d, np.nan)
    for i in range(len(high_1d)):
        if not (np.isnan(highest_high_senkou_b[i]) or np.isnan(lowest_low_senkou_b[i])):
            senkou_span_b[i] = (highest_high_senkou_b[i] + lowest_low_senkou_b[i]) / 2
    
    # Kumo Twist: Senkou Span A crosses Senkou Span B
    # Bullish twist: Senkou Span A crosses above Senkou Span B
    # Bearish twist: Senkou Span A crosses below Senkou Span B
    senkou_span_a_shift = np.roll(senkou_span_a, 1)
    senkou_span_b_shift = np.roll(senkou_span_b, 1)
    senkou_span_a_shift[0] = np.nan
    senkou_span_b_shift[0] = np.nan
    
    bullish_twist = (senkou_span_a > senkou_span_b) & (senkou_span_a_shift <= senkou_span_b_shift)
    bearish_twist = (senkou_span_a < senkou_span_b) & (senkou_span_a_shift >= senkou_span_b_shift)
    
    # Kumo Cloud: Future cloud (shifted 26 periods ahead)
    senkou_span_a_leading = np.roll(senkou_span_a, -period_kijun)
    senkou_span_b_leading = np.roll(senkou_span_b, -period_kijun)
    # Handle edge cases
    senkou_span_a_leading[-period_kijun:] = np.nan
    senkou_span_b_leading[-period_kijun:] = np.nan
    
    # Kumo Top/Bottom for current price comparison
    kumo_top = np.maximum(senkou_span_a, senkou_span_b)
    kumo_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # TK Cross: Tenkan-sen crosses Kijun-sen
    tenkan_shift = np.roll(tenkan_sen, 1)
    kijun_shift = np.roll(kijun_sen, 1)
    tenkan_shift[0] = np.nan
    kijun_shift[0] = np.nan
    
    tk_bullish_cross = (tenkan_sen > kijun_sen) & (tenkan_shift <= kijun_shift)
    tk_bearish_cross = (tenkan_sen < kijun_sen) & (tenkan_shift >= kijun_shift)
    
    # === Align Ichimoku components to 6h timeframe ===
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom)
    bullish_twist_aligned = align_htf_to_ltf(prices, df_1d, bullish_twist.astype(float))
    bearish_twist_aligned = align_htf_to_ltf(prices, df_1d, bearish_twist.astype(float))
    tk_bullish_cross_aligned = align_htf_to_ltf(prices, df_1d, tk_bullish_cross.astype(float))
    tk_bearish_cross_aligned = align_htf_to_ltf(prices, df_1d, tk_bearish_cross.astype(float))
    
    # === 6h Volume confirmation ===
    # Calculate 20-period average volume
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_confirm = volume > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(kumo_top_aligned[i]) or 
            np.isnan(kumo_bottom_aligned[i]) or
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long conditions:
            # 1. Price above Kumo (bullish bias)
            # 2. TK bullish cross OR Kumo bullish twist
            # 3. Volume confirmation
            price_above_kumo = close[i] > kumo_top_aligned[i]
            tk_bullish = tk_bullish_cross_aligned[i] > 0.5
            kumobull_twist = bullish_twist_aligned[i] > 0.5
            
            if price_above_kumo and (tk_bullish or kumobull_twist) and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
                continue
            
            # Short conditions:
            # 1. Price below Kumo (bearish bias)
            # 2. TK bearish cross OR Kumo bearish twist
            # 3. Volume confirmation
            price_below_kumo = close[i] < kumo_bottom_aligned[i]
            tk_bearish = tk_bearish_cross_aligned[i] > 0.5
            kumobear_twist = bearish_twist_aligned[i] > 0.5
            
            if price_below_kumo and (tk_bearish or kumobear_twist) and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Price drops below Kumo bottom OR TK bearish cross
            price_below_kumo_bottom = close[i] < kumo_bottom_aligned[i]
            tk_bearish_exit = tk_bearish_cross_aligned[i] > 0.5
            
            if price_below_kumo_bottom or tk_bearish_exit:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above Kumo top OR TK bullish cross
            price_above_kumo_top = close[i] > kumo_top_aligned[i]
            tk_bullish_exit = tk_bullish_cross_aligned[i] > 0.5
            
            if price_above_kumo_top or tk_bullish_exit:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_Trend_Filter_v1"
timeframe = "6h"
leverage = 1.0