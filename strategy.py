#!/usr/bin/env python3
# 6h_ichimoku_cloud_breakout_v1
# Hypothesis: 6h strategy using Ichimoku Cloud for trend identification and breakout confirmation.
# Long when price breaks above cloud in bullish regime (price > 1d EMA50); short when price breaks below cloud in bearish regime (price < 1d EMA50).
# Uses volume confirmation (>1.3x 20-bar average) to filter weak breakouts.
# Designed for low trade frequency (12-30/year) to minimize fee drag. Works in bull/bear via regime filter and cloud as dynamic support/resistance.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_breakout_v1"
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
    
    # Ichimoku Cloud components (9, 26, 52 periods)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = high_series.rolling(window=9, min_periods=9).max()
    period9_low = low_series.rolling(window=9, min_periods=9).min()
    tenkan_sen = ((period9_high + period9_low) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = high_series.rolling(window=26, min_periods=26).max()
    period26_low = low_series.rolling(window=26, min_periods=26).min()
    kijun_sen = ((period26_high + period26_low) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2, shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26 periods ahead
    period52_high = high_series.rolling(window=52, min_periods=52).max()
    period52_low = low_series.rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    # The cloud is between Senkou Span A and Senkou Span B
    # For plotting, these would be shifted forward, but for breakout detection we use current values
    # Upper cloud boundary = max(Senkou A, Senkou B)
    # Lower cloud boundary = min(Senkou A, Senkou B)
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Multi-timeframe: 1d EMA(50) for regime filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    ema_50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Regime filter: 1d EMA50 trend
        regime_uptrend = close[i] > ema_50_1d_aligned[i]
        regime_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below cloud (bearish cloud break)
            if close[i] < lower_cloud[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above cloud (bullish cloud break)
            if close[i] > upper_cloud[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for cloud breakout with volume and regime confirmation
            bullish_breakout = (close[i] > upper_cloud[i]) and volume_confirmed and regime_uptrend
            bearish_breakout = (close[i] < lower_cloud[i]) and volume_confirmed and regime_downtrend
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals