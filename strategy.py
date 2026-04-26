#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_Breakout_1dTrend_VolumeSpike
Hypothesis: Ichimoku cloud twist (Senkou Span A/B cross) on 6h with 1d EMA50 trend filter and volume confirmation. 
Cloud twist indicates momentum shift; combined with 1d trend and volume spike filters, it captures strong breakouts in both bull/bear markets.
Targets 50-150 total trades over 4 years via confluence requirements.
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
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou_b + low_senkou_b) / 2)
    
    # Cloud twist: Senkou Span A crosses above/below Senkou Span B
    # We need current and previous values to detect cross
    senkou_a_prev = np.roll(senkou_a, 1)
    senkou_b_prev = np.roll(senkou_b, 1)
    senkou_a_prev[0] = np.nan
    senkou_b_prev[0] = np.nan
    
    # Bullish twist: Senkou A crosses above Senkou B (previous A <= previous B and current A > current B)
    bullish_twist = (senkou_a_prev <= senkou_b_prev) & (senkou_a > senkou_b)
    # Bearish twist: Senkou A crosses below Senkou B (previous A >= previous B and current A < current B)
    bearish_twist = (senkou_a_prev >= senkou_b_prev) & (senkou_a < senkou_b)
    
    # Volume spike: volume > 1.5x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (1.5 * vol_median_20)
    
    # Fixed position size
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 52 for Senkou B, 26 for Kijun, 20 for volume median
    start_idx = max(52, 26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(tenkan_sen[i]) or
            np.isnan(kijun_sen[i]) or
            np.isnan(senkou_a[i]) or
            np.isnan(senkou_b[i]) or
            np.isnan(senkou_a_prev[i]) or
            np.isnan(senkou_b_prev[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Bullish: cloud twist bullish + volume spike + price above Kumo (close > max(Senkou A, Senkou B)) + uptrend (close > EMA50_1d)
            long_entry = bullish_twist[i] and vol_spike and (close_val > max(senkou_a[i], senkou_b[i])) and (close_val > ema_50_val)
            # Bearish: cloud twist bearish + volume spike + price below Kumo (close < min(Senkou A, Senkou B)) + downtrend (close < EMA50_1d)
            short_entry = bearish_twist[i] and vol_spike and (close_val < min(senkou_a[i], senkou_b[i])) and (close_val < ema_50_val)
            
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
            # Long - exit on trend reversal, price re-enters cloud, or opposite twist
            if close_val < ema_50_val or close_val < min(senkou_a[i], senkou_b[i]) or bearish_twist[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal, price re-enters cloud, or opposite twist
            if close_val > ema_50_val or close_val > max(senkou_a[i], senkou_b[i]) or bullish_twist[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0