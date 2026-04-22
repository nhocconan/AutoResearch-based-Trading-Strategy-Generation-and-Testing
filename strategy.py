#!/usr/bin/env python3
"""
Hypothesis: 6-hour Ichimoku Cloud with 1-day trend filter and volume confirmation.
Long when price is above Ichimoku cloud, Tenkan > Kijun, and 1-day EMA50 rising with volume spike.
Short when price is below Ichimoku cloud, Tenkan < Kijun, and 1-day EMA50 falling with volume spike.
Exit when price crosses Tenkan-Kijun line or 1-day EMA50 changes direction.
Ichimoku provides dynamic support/resistance and trend signals; 1-day EMA50 filters higher timeframe trend;
volume spike confirms institutional participation. Designed for moderate trade frequency by requiring
multiple confirmations and using 6h-level Ichimoku with daily trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku Cloud components."""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=tenkan).max()
    period9_low = pd.Series(low).rolling(window=tenkan).min()
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=kijun).max()
    period26_low = pd.Series(low).rolling(window=kijun).min()
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=senkou).max()
    period52_low = pd.Series(low).rolling(window=senkou).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Line): Close shifted 26 periods behind
    chikou_span = pd.Series(close).shift(-kijun)
    
    return tenkan_sen.values, kijun_sen.values, senkou_a.values, senkou_b.values, chikou_span.values

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Ichimoku on 6h data
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
    
    # 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after enough data for Ichimoku
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine if price is above or below cloud
        # Cloud top is the higher of senkou_a and senkou_b
        # Cloud bottom is the lower of senkou_a and senkou_b
        cloud_top = np.maximum(senkou_a[i], senkou_b[i])
        cloud_bottom = np.minimum(senkou_a[i], senkou_b[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price above cloud, Tenkan > Kijun, 1-day EMA50 rising, volume spike
            if (price_above_cloud and 
                tenkan[i] > kijun[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud, Tenkan < Kijun, 1-day EMA50 falling, volume spike
            elif (price_below_cloud and 
                  tenkan[i] < kijun[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below cloud OR Tenkan < Kijun OR 1-day EMA50 falls
                if (not price_above_cloud or 
                    tenkan[i] < kijun[i] or 
                    ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above cloud OR Tenkan > Kijun OR 1-day EMA50 rises
                if (not price_below_cloud or 
                    tenkan[i] > kijun[i] or 
                    ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_Cloud_1dEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0