#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Trend_1d_Strength
# Hypothesis: 6-hour Ichimoku cloud strategy with daily trend strength filter. Uses 1d ADX and DI crossover to confirm trend strength before taking Ichimoku signals. Avoids whipsaws in ranging markets by requiring ADX > 25 and DI+ > DI-. Works in both bull (trend following) and bear (avoids false signals) by filtering weak trends.

name = "6h_Ichimoku_Cloud_Trend_1d_Strength"
timeframe = "6h"
leverage = 1.0

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
    
    # Daily data for ADX trend strength filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX and DI on daily data
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value has no previous close
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        def smooth(val, period):
            smoothed = np.full_like(val, np.nan)
            if len(val) >= period:
                # Initial average
                smoothed[period-1] = np.nansum(val[:period])
                # Wilder smoothing
                for i in range(period, len(val)):
                    smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + val[i]
            return smoothed
        
        atr = smooth(tr, period)
        dm_plus_smooth = smooth(dm_plus, period)
        dm_minus_smooth = smooth(dm_minus, period)
        
        # DI+ and DI-
        di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
        di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
        
        # DX and ADX
        dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
        adx = smooth(dx, period)
        
        return adx, di_plus, di_minus
    
    adx_1d, di_plus_1d, di_minus_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Ichimoku components on 6h data
    def calculate_ichimoku(high, low, close):
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = np.maximum.accumulate(high)
        period9_low = np.minimum.accumulate(low)
        # For true rolling window, we need to adjust
        tenkan_sen = np.full_like(high, np.nan)
        kijun_sen = np.full_like(high, np.nan)
        senkou_span_a = np.full_like(high, np.nan)
        senkou_span_b = np.full_like(high, np.nan)
        
        # Calculate properly with rolling windows
        for i in range(len(high)):
            if i >= 8:  # 9 periods
                tenkan_sen[i] = (np.max(high[i-8:i+1]) + np.min(low[i-8:i+1])) / 2
            if i >= 26:  # 27 periods
                kijun_sen[i] = (np.max(high[i-26:i+1]) + np.min(low[i-26:i+1])) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
        for i in range(len(high)):
            if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
                if i + 26 < len(high):
                    senkou_span_a[i + 26] = (tenkan_sen[i] + kijun_sen[i]) / 2
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
        for i in range(len(high)):
            if i >= 51:  # 52 periods
                senkou_span_b[i + 26] = (np.max(high[i-51:i+1]) + np.min(low[i-51:i+1])) / 2
        
        # Chikou Span (Lagging Span): Close shifted 26 periods back
        chikou_span = np.full_like(close, np.nan)
        for i in range(26, len(close)):
            chikou_span[i-26] = close[i]
        
        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
    
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
    
    # Align daily ADX and DI to 6h timeframe (wait for 1d bar to close)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    di_plus_1d_aligned = align_htf_to_ltf(prices, df_1d, di_plus_1d)
    di_minus_1d_aligned = align_htf_to_ltf(prices, df_1d, di_minus_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough history for Ichimoku
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(di_plus_1d_aligned[i]) or 
            np.isnan(di_minus_1d_aligned[i]) or np.isnan(tenkan[i]) or 
            np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend strength filter: ADX > 25 and DI+ > DI- for uptrend, ADX > 25 and DI- > DI+ for downtrend
        strong_uptrend = adx_1d_aligned[i] > 25 and di_plus_1d_aligned[i] > di_minus_1d_aligned[i]
        strong_downtrend = adx_1d_aligned[i] > 25 and di_minus_1d_aligned[i] > di_plus_1d_aligned[i]
        
        # Ichimoku signals
        # Bullish: Price above cloud, Tenkan > Kijun, Chikou above price (26 periods back)
        price_above_cloud = close[i] > senkou_a[i] and close[i] > senkou_b[i]
        tenkan_above_kijun = tenkan[i] > kijun[i]
        chikou_above_price = False
        if i >= 26 and not np.isnan(chikou[i-26]):  # Chikou is shifted back
            chikou_above_price = chikou[i-26] > close[i-26]
        
        # Bearish: Price below cloud, Tenkan < Kijun, Chikou below price (26 periods back)
        price_below_cloud = close[i] < senkou_a[i] and close[i] < senkou_b[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        chikou_below_price = False
        if i >= 26 and not np.isnan(chikou[i-26]):
            chikou_below_price = chikou[i-26] < close[i-26]
        
        if position == 0:
            # Long: Strong uptrend + Ichimoku bullish signals
            if strong_uptrend and price_above_cloud and tenkan_above_kijun and chikou_above_price:
                signals[i] = 0.25
                position = 1
            # Short: Strong downtrend + Ichimoku bearish signals
            elif strong_downtrend and price_below_cloud and tenkan_below_kijun and chikou_below_price:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Trend weakens or Ichimoku turns bearish
            if not (strong_uptrend and price_above_cloud and tenkan_above_kijun):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Trend weakens or Ichimoku turns bullish
            if not (strong_downtrend and price_below_cloud and tenkan_below_kijun):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals