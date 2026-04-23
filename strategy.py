#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
Long when Tenkan > Kijun AND price above cloud AND close > 1d EMA50 AND volume > 1.5x 20-period average.
Short when Tenkan < Kijun AND price below cloud AND close < 1d EMA50 AND volume > 1.5x 20-period average.
Exit when Tenkan/Kijun cross reverses or price touches opposite cloud edge.
Ichimoku provides dynamic support/resistance and trend direction; 1d EMA50 filters higher timeframe trend; volume confirms conviction.
Works in bull (cloud acts as support) and bear (cloud acts as resistance) markets.
Targets 12-37 trades/year (50-150 over 4 years) with discrete sizing (0.25) to minimize fee drag.
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
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 50, 20)  # Ichimoku needs 52, EMA50 needs 50, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        ema50_val = ema50_1d_aligned[i]
        
        # Determine cloud boundaries (Senkou Span A/B shifted forward 26 periods)
        # For cloud at current period, we use Senkou values calculated 26 periods ago
        if i >= 26:
            senkou_a_current = senkou_a[i - 26]
            senkou_b_current = senkou_b[i - 26]
        else:
            # Not enough data for cloud, skip
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_current, senkou_b_current)
        cloud_bottom = min(senkou_a_current, senkou_b_current)
        
        if position == 0:
            # Long: Tenkan > Kijun AND price above cloud AND uptrend (close > EMA50) AND volume spike
            if tenkan[i] > kijun[i] and price > cloud_top and close[i] > ema50_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan < Kijun AND price below cloud AND downtrend (close < EMA50) AND volume spike
            elif tenkan[i] < kijun[i] and price < cloud_bottom and close[i] < ema50_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Tenkan/Kijun cross reverses
            if position == 1 and tenkan[i] <= kijun[i]:
                exit_signal = True
            elif position == -1 and tenkan[i] >= kijun[i]:
                exit_signal = True
            
            # Secondary exit: Price touches opposite cloud edge
            if position == 1 and price < cloud_bottom:
                exit_signal = True
            elif position == -1 and price > cloud_top:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Ichimoku_Cloud_1dEMA50_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0