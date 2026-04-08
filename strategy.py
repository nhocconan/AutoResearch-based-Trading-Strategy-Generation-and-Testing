#!/usr/bin/env python3
"""
6h Ichimoku Cloud with 1d Trend Filter and Volume Confirmation
Hypothesis: Ichimoku cloud signals on 6h timeframe, filtered by 1d EMA trend direction and volume spikes,
provide high-probability entries with controlled trade frequency. Works in bull markets via cloud breakouts
and in bear markets via cloud breakdowns. Targets 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = np.roll(close, 26)  # Will be handled in logic
    
    # Volume filter: current volume > 2.0x 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(tenkan_sen[i]) or 
            np.isnan(kijun_sen[i]) or
            np.isnan(senkou_a[i]) or
            np.isnan(senkou_b[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below cloud OR trend reverses
            cloud_top = max(senkou_a[i], senkou_b[i])
            cloud_bottom = min(senkou_a[i], senkou_b[i])
            if (close[i] < cloud_bottom or 
                close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above cloud OR trend reverses
            cloud_top = max(senkou_a[i], senkou_b[i])
            cloud_bottom = min(senkou_a[i], senkou_b[i])
            if (close[i] > cloud_top or 
                close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter: price vs 1d EMA50
            uptrend = close[i] > ema_50_1d_aligned[i]
            downtrend = close[i] < ema_50_1d_aligned[i]
            
            # Cloud values
            cloud_top = max(senkou_a[i], senkou_b[i])
            cloud_bottom = min(senkou_a[i], senkou_b[i])
            
            # Long: price breaks above cloud with uptrend and volume spike
            # Also require Tenkan > Kijun (bullish momentum)
            if (close[i] > cloud_top and 
                tenkan_sen[i] > kijun_sen[i] and 
                uptrend and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below cloud with downtrend and volume spike
            # Also require Tenkan < Kijun (bearish momentum)
            elif (close[i] < cloud_bottom and 
                  tenkan_sen[i] < kijun_sen[i] and 
                  downtrend and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals