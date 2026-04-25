#!/usr/bin/env python3
"""
6h Ichimoku Cloud + TK Cross + 1d ADX Trend Filter + Volume Spike
Hypothesis: Ichimoku cloud provides dynamic support/resistance and trend direction. TK cross signals momentum shifts. 
1d ADX > 25 filters for trending markets (works in both bull/bear regimes). Volume spike confirms breakout strength.
Designed for 6h timeframe with 50-150 total trades over 4 years. Uses discrete position sizing to minimize fee drag.
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
    
    # Get 1d data for ADX trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least 14 days for ADX
        return np.zeros(n)
    
    # Calculate 1d ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr14 = WilderSmooth(tr, period)
    dm_plus_14 = WilderSmooth(dm_plus, period)
    dm_minus_14 = WilderSmooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(tr14 != 0, 100 * dm_plus_14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus_14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = WilderSmooth(dx, period)
    
    # Align ADX to 6h timeframe (needs 2 extra 1d bars for confirmation like fractals)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx, additional_delay_bars=2)
    
    # Calculate Ichimoku components on 6h data
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low)/2
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    high_9 = rolling_max(high, 9)
    low_9 = rolling_min(low, 9)
    tenkan_sen = (high_9 + low_9) / 2
    
    # Base Line (Kijun-sen): (26-period high + 26-period low)/2
    high_26 = rolling_max(high, 26)
    low_26 = rolling_min(low, 26)
    kijun_sen = (high_26 + low_26) / 2
    
    # Leading Span A (Senkou Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # We'll use the current value (not shifted) for cloud calculation as we need current cloud
    
    # Leading Span B (Senkou Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = rolling_max(high, 52)
    low_52 = rolling_min(low, 52)
    senkou_span_b = (high_52 + low_52) / 2
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku (52), ADX, and volume MA
    start_idx = max(52, 34, 20)  # 52 for Ichimoku, 34 for ADX (14*2+6), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        adx_val = adx_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Ichimoku components
        tenkan = tenkan_sen[i]
        kijun = kijun_sen[i]
        span_a = senkou_span_a[i]
        span_b = senkou_span_b[i]
        
        # Cloud boundaries (top and bottom of cloud)
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # Trend filter: ADX > 25 indicates trending market
        is_trending = adx_val > 25
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # TK Cross: Tenkan-sen crossing Kijun-sen
        tk_cross_up = tenkan > kijun and tenkan_sen[i-1] <= kijun_sen[i-1]
        tk_cross_down = tenkan < kijun and tenkan_sen[i-1] >= kijun_sen[i-1]
        
        if position == 0:
            # Look for entry signals
            # Long: Price above cloud, TK cross up, trending market, volume confirmation
            long_signal = (curr_close > cloud_top) and tk_cross_up and is_trending and volume_confirm
            # Short: Price below cloud, TK cross down, trending market, volume confirmation
            short_signal = (curr_close < cloud_bottom) and tk_cross_down and is_trending and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: Price closes below cloud OR TK cross down
            if curr_close < cloud_bottom or tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price closes above cloud OR TK cross up
            if curr_close > cloud_top or tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_ADXTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0