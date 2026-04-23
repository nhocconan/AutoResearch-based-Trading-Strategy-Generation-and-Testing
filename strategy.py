#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
- Long when Tenkan-sen crosses above Kijun-sen AND price above Cloud AND 1d close > 1d EMA50 AND volume > 1.5x 20-period average
- Short when Tenkan-sen crosses below Kijun-sen AND price below Cloud AND 1d close < 1d EMA50 AND volume > 1.5x 20-period average
- Exit when Tenkan-sen crosses back opposite direction OR price exits Cloud in opposite direction
- Uses Ichimoku for trend/momentum/cloud support/resistance, proven effective in crypto
- 1d EMA50 filter ensures alignment with higher timeframe trend to avoid counter-trend trades
- Volume confirmation reduces false signals
- Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Ichimoku components (9, 26, 52 periods)
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
    
    # Cloud (Kumo) boundaries: Senkou Span A and B
    # Upper cloud = max(senkou_a, senkou_b)
    # Lower cloud = min(senkou_a, senkou_b)
    upper_cloud = np.maximum(senkou_a, senkou_b)
    lower_cloud = np.minimum(senkou_a, senkou_b)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 51, 20)  # Need 52 for Senkou B, 51 for EMA50 alignment, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Ichimoku signals
        tenkan_cross_above_kijun = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tenkan_cross_below_kijun = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        price_above_cloud = close[i] > upper_cloud[i]
        price_below_cloud = close[i] < lower_cloud[i]
        
        # Trend filter
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: bullish TK cross + price above cloud + uptrend + volume
            if tenkan_cross_above_kijun and price_above_cloud and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross + price below cloud + downtrend + volume
            elif tenkan_cross_below_kijun and price_below_cloud and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: bearish TK cross OR price below cloud
                if tenkan_cross_below_kijun or price_below_cloud:
                    exit_signal = True
            elif position == -1:
                # Exit short: bullish TK cross OR price above cloud
                if tenkan_cross_above_kijun or price_above_cloud:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Ichimoku_1dEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0