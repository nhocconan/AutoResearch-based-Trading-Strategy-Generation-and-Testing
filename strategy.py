#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
- Ichimoku (Tenkan/Kijun cross + price vs cloud) captures momentum and support/resistance.
- 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades.
- Volume > 1.3x 20-period average confirms breakout validity.
- Discrete position size 0.25 limits drawdown.
- Target: 15-30 trades/year on 6h timeframe (60-120 total over 4 years).
- Ichimoku works in both bull/bear regimes via cloud filtering and trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high9 + low9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high26 + low26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high52 + low52) / 2)
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    # Need 52 for Senkou B + 26 shift = 78 bars minimum
    start_idx = 78
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Cloud top and bottom (Senkou Span A and B)
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        if position == 0:
            # Long: Tenkan > Kijun AND price above cloud AND price above 1d EMA50 AND volume confirmation
            if (tenkan[i] > kijun[i] and 
                close[i] > cloud_top and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan < Kijun AND price below cloud AND price below 1d EMA50 AND volume confirmation
            elif (tenkan[i] < kijun[i] and 
                  close[i] < cloud_bottom and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Tenkan < Kijun OR price below cloud OR price below 1d EMA50
            if (tenkan[i] < kijun[i] or 
                close[i] < cloud_bottom or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Tenkan > Kijun OR price above cloud OR price above 1d EMA50
            if (tenkan[i] > kijun[i] or 
                close[i] > cloud_top or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0