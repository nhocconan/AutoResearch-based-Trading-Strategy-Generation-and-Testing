#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud Breakout with 1d Weekly Trend Filter and Volume Confirmation
# Uses Ichimoku (Tenkan-sen, Kijun-sen, Senkou Span A/B) from 6h data for entry signals
# 1d EMA50 determines higher timeframe trend bias (long only above, short only below)
# Volume spike (>1.8x 20-period EMA) confirms breakout authenticity
# Ichimoku works in all regimes: cloud acts as dynamic support/resistance, TK cross captures momentum
# Weekly trend filter prevents counter-trend trades during major reversals
# Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag

name = "6h_Ichimoku_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku calculations on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    highest_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (highest_high_tenkan + lowest_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    highest_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (highest_high_kijun + lowest_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    highest_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (highest_high_senkou_b + lowest_low_senkou_b) / 2
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid Ichimoku
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom (Senkou Span A/B)
        cloud_top = max(senkou_span_a[i], senkou_span_b[i])
        cloud_bottom = min(senkou_span_a[i], senkou_span_b[i])
        
        # Volume spike: current volume > 1.8 x 20-period EMA (balanced to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Ichimoku signals with 1d trend filter
        # Long: Price above cloud + Tenkan crosses above Kijun + price above 1d EMA50 + volume spike
        # Short: Price below cloud + Tenkan crosses below Kijun + price below 1d EMA50 + volume spike
        if position == 0:
            if (close[i] > cloud_top and tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1] and
                close[i] > ema_50_1d_aligned[i] and volume_spike):
                signals[i] = 0.25
                position = 1
            elif (close[i] < cloud_bottom and tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1] and
                  close[i] < ema_50_1d_aligned[i] and volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price falls below cloud OR Tenkan crosses below Kijun
            if close[i] < cloud_bottom or (tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price rises above cloud OR Tenkan crosses above Kijun
            if close[i] > cloud_top or (tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals