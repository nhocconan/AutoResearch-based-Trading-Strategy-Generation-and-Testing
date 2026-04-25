#!/usr/bin/env python3
"""
6h Ichimoku Cloud + 1d ADX Trend Filter + Volume Confirmation
Hypothesis: Ichimoku cloud acts as dynamic support/resistance, with TK cross providing momentum signals.
1d ADX > 25 ensures we only trade in strong trending markets (avoids chop). Volume confirmation filters weak breakouts.
Works in bull markets (cloud as support in uptrend) and bear markets (cloud as resistance in downtrend).
Target: 12-30 trades/year per symbol.
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
    
    # Volume confirmation: current volume > 1.8 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    # 1d ADX trend filter (MTF) - loaded ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Calculate ADX(14) on 1d data
    tr1 = np.maximum(df_1d_high - df_1d_low, np.abs(df_1d_high - np.roll(df_1d_close, 1)))
    tr2 = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr = np.maximum(tr1, tr2)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_dm = np.where((df_1d_high - np.roll(df_1d_high, 1)) > (np.roll(df_1d_low, 1) - df_1d_low),
                       np.maximum(df_1d_high - np.roll(df_1d_high, 1), 0), 0)
    minus_dm = np.where((np.roll(df_1d_low, 1) - df_1d_low) > (df_1d_high - np.roll(df_1d_high, 1)),
                        np.maximum(np.roll(df_1d_low, 1) - df_1d_low, 0), 0)
    
    plus_dm_smoothed = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smoothed = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * plus_dm_smoothed / atr
    minus_di = 100 * minus_dm_smoothed / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    strong_trend = adx_aligned > 25
    
    # Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # The cloud is between Senkou Span A and Senkou Span B
    # We use current cloud for support/resistance
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(52, 30) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or np.isnan(senkou_span_a[i]) or
            np.isnan(senkou_span_b[i]) or np.isnan(vol_ma[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        is_strong_trend = strong_trend[i]
        
        # Bullish TK cross: Tenkan crosses above Kijun
        tk_cross_bull = tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]
        # Bearish TK cross: Tenkan crosses below Kijun
        tk_cross_bear = tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]
        
        # Price above cloud (bullish) or below cloud (bearish)
        cloud_top = max(senkou_span_a[i], senkou_span_b[i])
        cloud_bottom = min(senkou_span_a[i], senkou_span_b[i])
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        if position == 0:
            # Look for entry signals - require: TK cross + volume spike + strong trend + price position relative to cloud
            long_entry = tk_cross_bull and vol_spike and is_strong_trend and price_above_cloud
            short_entry = tk_cross_bear and vol_spike and is_strong_trend and price_below_cloud
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when price crosses below cloud or TK cross turns bearish
            if price_below_cloud or (tenkan_sen[i] < kijun_sen[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above cloud or TK cross turns bullish
            if price_above_cloud or (tenkan_sen[i] > kijun_sen[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_ADXTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0