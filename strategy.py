#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d ADX trend filter and volume confirmation.
Long when price breaks above Ichimoku cloud (Senkou Span A/B) AND 1d ADX > 25 AND 6h volume > 1.5x 20-period MA.
Short when price breaks below Ichimoku cloud AND 1d ADX > 25 AND 6h volume > 1.5x 20-period MA.
Exit when price re-enters the cloud or 1d ADX falls below 20.
Uses 1d HTF for trend strength filter to avoid whipsaws, Ichimoku cloud as dynamic support/resistance,
volume confirmation to ensure momentum behind breakouts.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Ichimoku works in both bull (breakouts above cloud) and bear (breakdowns below cloud) markets.
ADX filter ensures we only trade when there is sufficient trend strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku components
    conversion_period = 9
    base_period = 26
    lagging_span_period = 52
    displacement = 26
    
    # Conversion Line (Tenkan-sen): (highest high + lowest low)/2 over past 9 periods
    highest_high_9 = pd.Series(high).rolling(window=conversion_period, min_periods=conversion_period).max().values
    lowest_low_9 = pd.Series(low).rolling(window=conversion_period, min_periods=conversion_period).min().values
    tenkan_sen = (highest_high_9 + lowest_low_9) / 2
    
    # Base Line (Kijun-sen): (highest high + lowest low)/2 over past 26 periods
    highest_high_26 = pd.Series(high).rolling(window=base_period, min_periods=base_period).max().values
    lowest_low_26 = pd.Series(low).rolling(window=base_period, min_periods=base_period).min().values
    kijun_sen = (highest_high_26 + lowest_low_26) / 2
    
    # Leading Span A (Senkou Span A): (Conversion Line + Base Line)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Leading Span B (Senkou Span B): (highest high + lowest low)/2 over past 52 periods shifted 26 periods ahead
    highest_high_52 = pd.Series(high).rolling(window=lagging_span_period, min_periods=lagging_span_period).max().values
    lowest_low_52 = pd.Series(low).rolling(window=lagging_span_period, min_periods=lagging_span_period).min().values
    senkou_span_b = ((highest_high_52 + lowest_low_52) / 2)
    
    # Calculate 1d ADX for trend strength filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def WilderMA(arr, period):
        """Wilder's Moving Average (same as RSI smoothing)"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr_period = 14
    atr = WilderMA(tr, atr_period)
    dm_plus_smooth = WilderMA(dm_plus, atr_period)
    dm_minus_smooth = WilderMA(dm_minus, atr_period)
    
    # Directional Indicators
    di_plus = np.where(atr != 0, dm_plus_smooth / atr * 100, 0)
    di_minus = np.where(atr != 0, dm_minus_smooth / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = WilderMA(dx, atr_period)
    
    # Align HTF data to LTF
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(
        conversion_period, base_period, lagging_span_period,  # Ichimoku
        atr_period * 2,  # ADX needs double smoothing
        20  # volume MA
    ) + displacement  # Add displacement for Senkou Span
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        # Cloud boundaries (Senkou Span A/B) - note: already displaced in calculation
        upper_cloud = max(senkou_span_a[i], senkou_span_b[i])
        lower_cloud = min(senkou_span_a[i], senkou_span_b[i])
        
        # Aligned cloud boundaries for current bar (already accounts for displacement)
        upper_cloud_aligned = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud_aligned = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Volume filter: 6h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Price breaks above cloud AND ADX > 25 (strong trend) AND volume filter
            if price > upper_cloud_aligned and adx_val > 25 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below cloud AND ADX > 25 (strong trend) AND volume filter
            elif price < lower_cloud_aligned and adx_val > 25 and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Price re-enters cloud OR ADX falls below 20 (trend weakening)
                if price < upper_cloud_aligned or adx_val < 20:
                    exit_signal = True
            elif position == -1:
                # Short exit: Price re-enters cloud OR ADX falls below 20 (trend weakening)
                if price > lower_cloud_aligned or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_Cloud_Breakout_1dADX_Trend_Volume"
timeframe = "6h"
leverage = 1.0