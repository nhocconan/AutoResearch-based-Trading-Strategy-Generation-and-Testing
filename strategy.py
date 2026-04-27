#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_VolumeSpike_HTF
Hypothesis: Ichimoku cloud twist (Senkou Span A/B cross) on 6h with 1d EMA50 trend filter and volume confirmation.
Cloud twist indicates potential trend change. Trading only in direction of 1d trend avoids counter-trend whipsaws.
Volume spike confirms breakout authenticity. Designed for 6h timeframe with target 50-150 trades over 4 years.
Uses discrete position sizing (0.25) to minimize fee churn while maintaining adequate exposure.
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
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind
    # Not used for signals as it requires future data
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for Ichimoku (52), EMA50, volume average
    start_idx = max(100, 52, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        size = 0.25  # 25% position size
        
        # Kumo twist detection: Senkou A crosses Senkou B
        # Bullish twist: Senkou A crosses above Senkou B
        # Bearish twist: Senkou A crosses below Senkou B
        if i >= 1:
            bullish_twist = senkou_a[i] > senkou_b[i] and senkou_a[i-1] <= senkou_b[i-1]
            bearish_twist = senkou_a[i] < senkou_b[i] and senkou_a[i-1] >= senkou_b[i-1]
        else:
            bullish_twist = False
            bearish_twist = False
        
        if position == 0:
            # Flat - look for entry: Kumo twist in direction of 1d trend with volume spike
            # Long: bullish twist AND 1d trend is up (close > EMA50) AND volume spike
            # Short: bearish twist AND 1d trend is down (close < EMA50) AND volume spike
            if bullish_twist and close_val > ema_trend and vol_spike:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif bearish_twist and close_val < ema_trend and vol_spike:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when Kumo twist turns bearish or price re-enters cloud (Senkou A < Senkou B)
            if bearish_twist or senkou_a[i] < senkou_b[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when Kumo twist turns bullish or price re-enters cloud (Senkou A > Senkou B)
            if bullish_twist or senkou_a[i] > senkou_b[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_VolumeSpike_HTF"
timeframe = "6h"
leverage = 1.0