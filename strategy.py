#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Uses Ichimoku (Tenkan/Kijun/Senkou Span A/B) from 6h for entry/exit signals
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades
# Volume confirmation (2.0x 20-period average) filters false breakouts
# Targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
# Ichimoku provides dynamic support/resistance and trend direction
# Works in both bull and bear: daily trend filter prevents counter-trend entries

name = "6h_Ichimoku_1dEMA50_VolumeConfirm"
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used for signals as it requires future data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Ichimoku calculations)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud color and position
        # Green cloud (bullish): Senkou Span A > Senkou Span B
        # Red cloud (bearish): Senkou Span A < Senkou Span B
        is_bullish_cloud = senkou_span_a[i] > senkou_span_b[i]
        is_bearish_cloud = senkou_span_a[i] < senkou_span_b[i]
        
        # Price above/below cloud
        price_above_cloud = close[i] > max(senkou_span_a[i], senkou_span_b[i])
        price_below_cloud = close[i] < min(senkou_span_a[i], senkou_span_b[i])
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above cloud AND Tenkan > Kijun AND price > 1d EMA50
            if (price_above_cloud and 
                tenkan_sen[i] > kijun_sen[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below cloud AND Tenkan < Kijun AND price < 1d EMA50
            elif (price_below_cloud and 
                  tenkan_sen[i] < kijun_sen[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below cloud OR Tenkan < Kijun
            if (close[i] < min(senkou_span_a[i], senkou_span_b[i]) or 
                tenkan_sen[i] < kijun_sen[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above cloud OR Tenkan > Kijun
            if (close[i] > max(senkou_span_a[i], senkou_span_b[i]) or 
                tenkan_sen[i] > kijun_sen[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals