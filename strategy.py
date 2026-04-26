#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_VolumeFilter
Hypothesis: 6h Ichimoku cloud twist (Tenkan/Kijun cross) with 1d trend filter and volume confirmation.
Enters long when Tenkan crosses above Kijun AND price is above cloud (bullish twist) with 1d uptrend and volume spike.
Enters short when Tenkan crosses below Kijun AND price is below cloud (bearish twist) with 1d downtrend and volume spike.
Exits on opposite twist or trend failure.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year on 6h.
Works in bull/bear by aligning with 1d trend to avoid counter-trend trades.
Ichimoku twist captures momentum shifts while cloud acts as dynamic support/resistance.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 26+26 for Ichimoku
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter (more responsive than 34)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2.0
    
    # Volume confirmation: volume > 1.8x 30-period MA
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Determine Ichimoku twist (Tenkan/Kijun cross)
        bullish_twist = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        bearish_twist = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Determine price vs cloud
        above_cloud = close[i] > senkou_a[i] and close[i] > senkou_b[i]
        below_cloud = close[i] < senkou_a[i] and close[i] < senkou_b[i]
        
        if position == 0:
            # Long: bullish twist + above cloud + 1d uptrend + volume spike
            if (bullish_twist and above_cloud and 
                close[i] > ema_50_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish twist + below cloud + 1d downtrend + volume spike
            elif (bearish_twist and below_cloud and 
                  close[i] < ema_50_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: bearish twist OR price drops below cloud OR 1d trend turns bearish
            if (bearish_twist or below_cloud or close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: bullish twist OR price rises above cloud OR 1d trend turns bullish
            if (bullish_twist or above_cloud or close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0