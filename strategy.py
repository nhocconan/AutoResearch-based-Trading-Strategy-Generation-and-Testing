#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Ichimoku cloud direction with 6h volume confirmation and ADX trend filter.
# Long when price > 1d Ichimoku cloud (Senkou Span A/B), 6h ADX > 25 (trending), and volume > 1.5x 20-period average.
# Short when price < 1d Ichimoku cloud, 6h ADX > 25, and volume > 1.5x 20-period average.
# Exit when price crosses back into the cloud or ADX drops below 20 (trend weakening).
# Uses discrete position size 0.25. Ichimoku cloud provides strong support/resistance from higher timeframe,
# ADX filters for trending conditions only, volume confirms momentum. Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Ichimoku Cloud ===
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (max_high_senkou_b + min_low_senkou_b) / 2.0
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Get 6h data once before loop for ADX and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === 6h ADX (Average Directional Index) ===
    # True Range
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_6h - np.roll(high_6h, 1)) > (np.roll(low_6h, 1) - low_6h),
                       np.maximum(high_6h - np.roll(high_6h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_6h, 1) - low_6h) > (high_6h - np.roll(high_6h, 1)),
                        np.maximum(np.roll(low_6h, 1) - low_6h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[period-1] = np.nansum(values[:period])  # first value is simple average
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    period_adx = 14
    atr = wilders_smoothing(tr, period_adx)
    dm_plus_smooth = wilders_smoothing(dm_plus, period_adx)
    dm_minus_smooth = wilders_smoothing(dm_minus, period_adx)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / np.where(atr != 0, atr, np.nan)
    di_minus = 100 * dm_minus_smooth / np.where(atr != 0, atr, np.nan)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) != 0, (di_plus + di_minus), np.nan)
    adx = wilders_smoothing(dx, period_adx)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_6h, adx)
    
    # Volume moving average (20-period) on 6h
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20_6h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        senkou_a = senkou_span_a_aligned[i]
        senkou_b = senkou_span_b_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Determine cloud boundaries (top and bottom of cloud)
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below cloud bottom or ADX drops below 20 (trend weakening)
            if price < cloud_bottom or adx_val < 20:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above cloud top or ADX drops below 20 (trend weakening)
            if price > cloud_top or adx_val < 20:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Trend filter: ADX > 25 (strong trend)
            trend_filter = adx_val > 25
            
            # Volume filter: volume > 1.5x 20-period average
            vol_filter = vol > 1.5 * vol_ma_val
            
            # Price filter: price relative to cloud
            price_above_cloud = price > cloud_top
            price_below_cloud = price < cloud_bottom
            
            # LONG: price above cloud, strong trend, volume confirmation
            if price_above_cloud and trend_filter and vol_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: price below cloud, strong trend, volume confirmation
            elif price_below_cloud and trend_filter and vol_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_1dIchimoku_Cloud_ADX_VolumeFilter_V1"
timeframe = "6h"
leverage = 1.0