#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with weekly ADX trend filter and volume confirmation.
# Ichimoku provides dynamic support/resistance via Kumo (cloud) and momentum via TK cross.
# Weekly ADX ensures we only trade in strong trends, avoiding whipsaws in ranging markets.
# Volume confirmation adds conviction to breakouts.
# Designed for low trade frequency (15-35/year) to minimize fee drag in 6h timeframe.
# Works in bull markets (bullish TK cross above cloud with rising ADX) and bear markets 
# (bearish TK cross below cloud with rising ADX).
name = "6h_Ichimoku_TK_Cross_WeeklyADX_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    # Get weekly data for ADX filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Ichimoku components (using previous day's data to avoid look-ahead)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = df_1d['high'].rolling(window=9, min_periods=9).max().shift(1).values
    low_9 = df_1d['low'].rolling(window=9, min_periods=9).min().shift(1).values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = df_1d['high'].rolling(window=26, min_periods=26).max().shift(1).values
    low_26 = df_1d['low'].rolling(window=26, min_periods=26).min().shift(1).values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = df_1d['high'].rolling(window=52, min_periods=52).max().shift(1).values
    low_52 = df_1d['low'].rolling(window=52, min_periods=52).min().shift(1).values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate weekly ADX for trend strength
    # ADX calculation requires +DI and -DI
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # True Range
    tr1 = high_w[1:] - low_w[1:]
    tr2 = np.abs(high_w[1:] - close_w[:-1])
    tr3 = np.abs(low_w[1:] - close_w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_w[1:] - high_w[:-1]) > (low_w[:-1] - low_w[1:]), 
                       np.maximum(high_w[1:] - high_w[:-1], 0), 0)
    dm_minus = np.where((low_w[:-1] - low_w[1:]) > (high_w[1:] - high_w[:-1]), 
                        np.maximum(low_w[:-1] - low_w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (using Wilder's smoothing, equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nansum(data[:period]) / period
            # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    result[i] = np.nan
        return result
    
    atr_period = 14
    atr = wilders_smoothing(tr, atr_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, atr_period)
    
    # DI values
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, atr_period)  # ADX is smoothed DX
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: TK cross bullish (Tenkan > Kijun) AND price above cloud AND strong trend (ADX > 25) AND volume
            tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
            price_above_cloud = close[i] > cloud_top
            strong_trend = adx_aligned[i] > 25
            
            if vol_confirm and tk_bullish and price_above_cloud and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish (Tenkan < Kijun) AND price below cloud AND strong trend (ADX > 25) AND volume
            elif (vol_confirm and 
                  tenkan_aligned[i] < kijun_aligned[i] and 
                  close[i] < cloud_bottom and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TK cross bearish OR price falls below cloud bottom
            tk_bearish = tenkan_aligned[i] < kijun_aligned[i]
            price_below_cloud = close[i] < cloud_bottom
            
            if tk_bearish or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TK cross bullish OR price rises above cloud top
            tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
            price_above_cloud = close[i] > cloud_top
            
            if tk_bullish or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals