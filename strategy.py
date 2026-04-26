#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_VolumeSpike
Hypothesis: Ichimoku Tenkan-Kijun cross on 6h with 1d trend filter (price > 1d EMA50 for long, < for short) and volume confirmation (1.5x average volume). 
Only takes trades when price is above/below the Kumo (cloud) to ensure trend alignment. 
Designed for moderate trade frequency (12-30/year) to avoid fee drag on 6h timeframe.
Uses discrete position sizing (0.25) to minimize churn.
Works in both bull and bear markets by requiring trend alignment via 1d EMA50 and cloud filter.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind (not used for entry)
    
    # The Kumo (cloud) is between Senkou Span A and Senkou Span B
    # For trend: price above cloud = bullish, price below cloud = bearish
    # We need to compare current price with the cloud that was plotted 26 periods ago
    # So we shift senkou_a and senkou_b BACK by 26 to align with current price
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # Fill the first 26 values with NaN since they don't have valid cloud data
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Ichimoku components (52), 1d EMA (50), volume MA (20)
    start_idx = max(52, 50, 20) + 26  # +26 for the cloud shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a_shifted[i]) or np.isnan(senkou_b_shifted[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
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
        close_val = close[i]
        senkou_a_val = senkou_a_shifted[i]
        senkou_b_val = senkou_b_shifted[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = max(senkou_a_val, senkou_b_val)
        lower_cloud = min(senkou_a_val, senkou_b_val)
        
        # Trend filters
        price_above_close = close_val > upper_cloud
        price_below_close = close_val < lower_cloud
        
        # 1d trend filter
        uptrend_1d = close_val > ema_50_1d_val
        downtrend_1d = close_val < ema_50_1d_val
        
        # Volume confirmation
        volume_spike = volume_val > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Tenkan crosses above Kijun, price above cloud, 1d uptrend, volume spike
            tk_cross_up = tenkan_val > kijun_val and tenkan[i-1] <= kijun[i-1]
            long_signal = tk_cross_up and price_above_close and uptrend_1d and volume_spike
            
            # Short: Tenkan crosses below Kijun, price below cloud, 1d downtrend, volume spike
            tk_cross_down = tenkan_val < kijun_val and tenkan[i-1] >= kijun[i-1]
            short_signal = tk_cross_down and price_below_close and downtrend_1d and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Tenkan crosses below Kijun OR price breaks below cloud OR 1d trend turns down
            tk_cross_down = tenkan_val < kijun_val and tenkan[i-1] >= kijun[i-1]
            if tk_cross_down or (close_val < lower_cloud) or (close_val < ema_50_1d_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Tenkan crosses above Kijun OR price breaks above cloud OR 1d trend turns up
            tk_cross_up = tenkan_val > kijun_val and tenkan[i-1] <= kijun[i-1]
            if tk_cross_up or (close_val > upper_cloud) or (close_val > ema_50_1d_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0