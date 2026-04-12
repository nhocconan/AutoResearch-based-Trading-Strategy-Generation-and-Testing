#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_ichimoku_cloud_trend_v2
# Uses daily Ichimoku cloud (Tenkan-sen, Kijun-sen, Senkou Span A/B) for trend direction.
# Long when price > Kumo (cloud) and Tenkan > Kijun; short when price < Kumo and Tenkan < Kijun.
# Filters trades with 6h ADX > 20 to avoid ranging markets.
# Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drag.
# Ichimoku works in both bull (trend following) and bear (trend continuation) markets.

name = "6h_1d_ichimoku_cloud_trend_v2"
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
    
    # Get daily data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need at least 52 days for Senkou Span B
        return np.zeros(n)
    
    # Ichimoku components (daily)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_52 + min_low_52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Kumo (cloud) boundaries: Senkou Span A and B
    # Note: Senkou Span is plotted 26 periods ahead, so we need to shift back for current cloud
    # But since we're using align_htf_to_ltf which already handles the delay, we use current values
    # The cloud is between senkou_a and senkou_b
    
    # ADX trend filter on 6h timeframe
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        
        # Wilder's smoothing
        def wilders_smooth(data, period):
            result = np.full_like(data, np.nan, dtype=float)
            if len(data) < period:
                return result
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            return result
        
        atr = wilders_smooth(tr, period)
        plus_dm_smooth = wilders_smooth(plus_dm, period)
        minus_dm_smooth = wilders_smooth(minus_dm, period)
        
        plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
        minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilders_smooth(dx, period)
        return adx
    
    adx_6h = calculate_adx(high, low, close, 14)
    adx_filter = adx_6h > 20  # trend filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # start after warmup
        # Skip if Ichimoku levels not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or 
            np.isnan(adx_filter[i])):
            signals[i] = 0.0
            continue
        
        # Skip if no trend
        if not adx_filter[i]:
            # Hold current position if no trend
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = min(senkou_a_6h[i], senkou_b_6h[i])
        
        # Long conditions: price above cloud AND Tenkan > Kijun (bullish)
        if close[i] > cloud_top and tenkan_6h[i] > kijun_6h[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short conditions: price below cloud AND Tenkan < Kijun (bearish)
        elif close[i] < cloud_bottom and tenkan_6h[i] < kijun_6h[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite cloud cross
        elif close[i] < cloud_bottom and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > cloud_top and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals