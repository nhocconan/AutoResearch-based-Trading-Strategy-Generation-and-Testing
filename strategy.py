#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d ADX trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for ADX trend strength.
- Ichimoku components: Tenkan-sen (9-period), Kijun-sen (26-period), Senkou Span A/B (52-period displaced).
- Breakout: Price breaks above/below cloud with Tenkan/Kijun cross in direction of breakout.
- Trend filter: Only trade when 1d ADX > 25 (strong trend) to avoid whipsaws in ranging markets.
- Volume confirmation: Current volume > 1.5x 20-period volume MA to confirm breakout strength.
- Works in bull via buying cloud breakouts in uptrend, in bear via selling cloud breakdowns in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX for trend strength filter
    # ADX calculation requires +DI, -DI, and DX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    atr = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # +DI and -DI
    plus_di = 100 * dm_plus_smooth / atr
    minus_di = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period)
    
    # Align 1d ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Ichimoku calculations on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    kijun_sen = (pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(-period_kijun)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = ((pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                      pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2).shift(-period_kijun)
    
    # Current cloud boundaries (Senkou Span A/B from 26 periods ago)
    senkou_span_a_lagged = senkou_span_a.shift(period_kijun)  # Now aligned with current price
    senkou_span_b_lagged = senkou_span_b.shift(period_kijun)  # Now aligned with current price
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_lagged.values, senkou_span_b_lagged.values)
    cloud_bottom = np.minimum(senkou_span_a_lagged.values, senkou_span_b_lagged.values)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(period_kijun + period_senkou_b, 30)  # Ichimoku + ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen.iloc[i]) or np.isnan(kijun_sen.iloc[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Ichimoku cloud breakout with volume spike and ADX trend filter
            if volume_spike[i] and adx_aligned[i] > 25:  # Strong trend filter
                # Bullish conditions: price above cloud AND Tenkan > Kijun
                bullish = (close[i] > cloud_top[i]) and (tenkan_sen.iloc[i] > kijun_sen.iloc[i])
                # Bearish conditions: price below cloud AND Tenkan < Kijun
                bearish = (close[i] < cloud_bottom[i]) and (tenkan_sen.iloc[i] < kijun_sen.iloc[i])
                
                if bullish:
                    signals[i] = 0.25
                    position = 1
                elif bearish:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price re-enters cloud or opposite signal
            if close[i] < cloud_top[i]:  # Exit when price falls below cloud top
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters cloud or opposite signal
            if close[i] > cloud_bottom[i]:  # Exit when price rises above cloud bottom
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dADX_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0