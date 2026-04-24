#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
- Long when Tenkan > Kijun (bullish TK cross) AND price above Kumo (cloud) AND close > 1d EMA50 (bullish trend) AND volume > 1.5 * median volume
- Short when Tenkan < Kijun (bearish TK cross) AND price below Kumo (cloud) AND close < 1d EMA50 (bearish trend) AND volume > 1.5 * median volume
- Exit on opposite TK cross or trend reversal (close crosses 1d EMA50)
- Uses 6h primary timeframe with 1d HTF to target 50-150 total trades over 4 years (12-37/year)
- Ichimoku provides dynamic support/resistance via cloud and momentum via TK cross
- 1d EMA50 ensures alignment with higher timeframe trend to avoid whipsaws
- Volume confirmation reduces false signals
- Designed for BTC/ETH with edge in both trending and ranging markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    highest_tenkan = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_tenkan = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    highest_kijun = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_kijun = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_span_a = (tenkan + kijun) / 2
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods plotted 26 periods ahead
    highest_senkou_b = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_senkou_b = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = (highest_senkou_b + lowest_senkou_b) / 2
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 1.5 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_filter = volume > (1.5 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period, displacement) + displacement + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_span_a[i]) or 
            np.isnan(senkou_span_b[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B shifted forward)
        # For current price, we need cloud values that were plotted displacement periods ago
        cloud_top = max(senkou_span_a[i - displacement], senkou_span_b[i - displacement]) if i >= displacement else np.nan
        cloud_bottom = min(senkou_span_a[i - displacement], senkou_span_b[i - displacement]) if i >= displacement else np.nan
        
        if np.isnan(cloud_top) or np.isnan(cloud_bottom):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish TK cross, price above cloud, bullish trend, volume filter
            if (tenkan[i] > kijun[i] and 
                close[i] > cloud_top and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross, price below cloud, bearish trend, volume filter
            elif (tenkan[i] < kijun[i] and 
                  close[i] < cloud_bottom and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish TK cross OR price below cloud OR trend reversal
            if (tenkan[i] < kijun[i] or 
                close[i] < cloud_bottom or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish TK cross OR price above cloud OR trend reversal
            if (tenkan[i] > kijun[i] or 
                close[i] > cloud_top or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_IchimokuTK_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0