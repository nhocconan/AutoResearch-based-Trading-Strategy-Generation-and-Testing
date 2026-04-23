#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
- Primary timeframe: 6h, HTF: 1d for trend filter and cloud calculation
- Long: Price > Kumo (cloud) + Tenkan > Kijun + price > 1d EMA50 (uptrend) + volume > 1.5x 20-period avg
- Short: Price < Kumo (cloud) + Tenkan < Kijun + price < 1d EMA50 (downtrend) + volume > 1.5x 20-period avg
- Exit: Price crosses Tenkan-Kijun line (TK cross in opposite direction)
- Uses Ichimoku cloud as dynamic support/resistance with TK cross for momentum
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to balance return and risk
- BTC/ETH focus: requires HTF trend alignment to avoid SOL-only bias
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.5x 20-period average (volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Ichimoku components from 1d data
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2.0)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    high_senkou = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou + low_senkou) / 2.0)
    
    # Align Ichimoku components to 6h timeframe (values from 1d chart)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    # Need 52 for Senkou B, 26 for Kijun, 9 for Tenkan, 50 for EMA, 20 for volume MA
    start_idx = max(52, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Kumo (cloud) boundaries: Senkou Span A and Senkou Span B
        # Top of cloud is max(Senkou A, Senkou B), bottom is min(Senkou A, Senkou B)
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # TK cross: Tenkan > Kijun = bullish, Tenkan < Kijun = bearish
        tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
        tk_bearish = tenkan_aligned[i] < kijun_aligned[i]
        
        if position == 0:
            # Long: Price > Kumo (cloud) + Tenkan > Kijun + price > 1d EMA50 + volume spike
            if (close[i] > cloud_top and 
                tk_bullish and 
                close[i] > ema_50_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price < Kumo (cloud) + Tenkan < Kijun + price < 1d EMA50 + volume spike
            elif (close[i] < cloud_bottom and 
                  tk_bearish and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below Kumo OR Tenkan crosses below Kijun (TK cross bearish)
            if (close[i] < cloud_top or 
                tk_bearish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above Kumo OR Tenkan crosses above Kijun (TK cross bullish)
            if (close[i] > cloud_bottom or 
                tk_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0