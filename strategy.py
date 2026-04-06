#!/usr/bin/env python3
"""
6h Ichimoku Cloud strategy with 12h trend filter and volume confirmation.
Hypothesis: Ichimoku cloud acts as dynamic support/resistance; TK cross signals momentum shifts.
Using 12h trend filter (price above/below cloud) reduces false signals. Volume confirms breakout strength.
Works in bull via cloud breakouts and in bear via cloud breakdowns. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14259_6h_ichimoku_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components with proper min_periods"""
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over tenkan period
    tenkan_sen = (pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max() + 
                  pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over kijun period
    kijun_sen = (pd.Series(high).rolling(window=kijun, min_periods=kijun).max() + 
                 pd.Series(low).rolling(window=kijun, min_periods=kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted forward kijun periods
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over senkou period shifted forward kijun
    senkou_b = ((pd.Series(high).rolling(window=senkou, min_periods=senkou).max() + 
                 pd.Series(low).rolling(window=senkou, min_periods=senkou).min()) / 2).shift(kijun)
    
    # Chikou Span (Lagging Span): close shifted back kijun periods
    chikou_span = pd.Series(close).shift(-kijun)
    
    return tenkan_sen.values, kijun_sen.values, senkou_a.values, senkou_b.values, chikou_span.values

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Ichimoku for trend filter
    tenkan_12h, kijun_12h, senkou_a_12h, senkou_b_12h, _ = calculate_ichimoku(high_12h, low_12h, close_12h)
    # Determine trend: price above/below cloud
    # Cloud top = max(senkou_a, senkou_b), cloud bottom = min(senkou_a, senkou_b)
    cloud_top_12h = np.maximum(senkou_a_12h, senkou_b_12h)
    cloud_bottom_12h = np.minimum(senkou_a_12h, senkou_b_12h)
    trend_up_12h = close_12h > cloud_top_12h      # Bullish trend: price above cloud
    trend_down_12h = close_12h < cloud_bottom_12h # Bearish trend: price below cloud
    trend_up_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_up_12h)
    trend_down_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_down_12h)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h Ichimoku for entry signals
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.8 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of senkou period for Ichimoku, 20 for volume, 14 for ATR)
    start = max(52, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or \
           np.isnan(atr[i]) or np.isnan(vol_ma[i]) or np.isnan(trend_up_12h_aligned[i]) or np.isnan(trend_down_12h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Ichimoku signals with 12h trend filter and volume
        # Cloud top/bottom for current period
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        # TK Cross signals
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Long: TK cross up + price above cloud + 12h bullish trend + volume
        # Short: TK cross down + price below cloud + 12h bearish trend + volume
        long_signal = tk_cross_up and (close[i] > cloud_top) and trend_up_12h_aligned[i] and vol_filter[i]
        short_signal = tk_cross_down and (close[i] < cloud_bottom) and trend_down_12h_aligned[i] and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or TK cross down
            if close[i] <= stop_price or tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or TK cross up
            if close[i] >= stop_price or tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals