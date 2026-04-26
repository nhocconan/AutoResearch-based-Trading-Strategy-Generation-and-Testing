#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_v1
Hypothesis: Trade 6h Ichimoku cloud breakouts with 1d weekly bias filter and volume confirmation.
Uses Ichimoku (Tenkan=9, Kijun=26, Senkou B=52) from 6h data, aligned 1d EMA50 for trend,
and 1.5x volume spike. Long when price breaks above cloud in uptrend, short when breaks below cloud in downtrend.
Position size 0.25. Designed to capture strong trends while avoiding chop via cloud filter and EMA50.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
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
    
    # Get 1d data for HTF trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Current cloud boundaries: Senkou A and B shifted forward 26 periods
    # For breakout detection, we use current Senkou A/B (not shifted) as cloud edges
    # Upper cloud = max(Senkou A, Senkou B)
    # Lower cloud = min(Senkou A, Senkou B)
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    # Volume confirmation: 1.5x median volume
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Warmup: max of Ichimoku calculations (52), 1d EMA (50), volume median (30)
    start_idx = max(52, 50, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_median[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        senkou_a_val = senkou_a[i]
        senkou_b_val = senkou_b[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        upper_cloud_val = upper_cloud[i]
        lower_cloud_val = lower_cloud[i]
        
        if position == 0:
            # Long: price breaks above upper cloud, uptrend (close > EMA50), volume spike
            long_signal = (close_val > upper_cloud_val) and \
                          (close_val > ema_50_1d_val) and \
                          (volume_val > 1.5 * vol_median_val)
            # Short: price breaks below lower cloud, downtrend (close < EMA50), volume spike
            short_signal = (close_val < lower_cloud_val) and \
                           (close_val < ema_50_1d_val) and \
                           (volume_val > 1.5 * vol_median_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long with minimum holding period
            bars_since_entry += 1
            signals[i] = 0.25
            # Exit: price re-enters cloud (close < Senkou A) or trend reversal (close < EMA50) after minimum holding period
            if bars_since_entry >= 4 and ((close_val < senkou_a_val) or (close_val < ema_50_1d_val)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short with minimum holding period
            bars_since_entry += 1
            signals[i] = -0.25
            # Exit: price re-enters cloud (close > Senkou B) or trend reversal (close > EMA50) after minimum holding period
            if bars_since_entry >= 4 and ((close_val > senkou_b_val) or (close_val > ema_50_1d_val)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_v1"
timeframe = "6h"
leverage = 1.0