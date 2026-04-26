#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_v1
Hypothesis: Trade Ichimoku cloud breaks on 6h with 1d trend filter (price >/ < Kumo + TK cross).
Only take longs when price above cloud and TK cross bullish in uptrend (price > 1d EMA50),
and shorts when price below cloud and TK cross bearish in downtrend (price < 1d EMA50).
Ichimoku provides objective support/resistance (cloud) and momentum (TK cross).
Works in bull/bear by following 1d EMA50 trend and filtering counter-trend cloud breaks.
Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years).
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind (not used for signals)
    
    # Align HTF indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # For Ichimoku, we need to align the plotted Senkou spans (they are already forward-shifted)
    # But since we calculate them on 6h directly, no alignment needed for the indicator values themselves
    # However, we do need to ensure we don't use future data - the Senkou spans are plotted ahead
    # so we use the current values which represent future cloud - this is correct for Ichimoku
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Tenkan (9), Kijun (26), Senkou B (52), 1d EMA (50)
    start_idx = max(period_tenkan, period_kijun, period_senkou_b, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        close_val = close[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        
        # Cloud boundaries (Senkou Span A and B)
        upper_cloud = max(senkou_a[i], senkou_b[i])
        lower_cloud = min(senkou_a[i], senkou_b[i])
        
        # TK Cross
        tk_bullish = tenkan[i] > kijun[i]
        tk_bearish = tenkan[i] < kijun[i]
        
        if position == 0:
            # Long: price above cloud, TK bullish, and uptrend (close > 1d EMA50)
            long_signal = (close_val > upper_cloud) and \
                          tk_bullish and \
                          (close_val > ema_50_1d_val)
            
            # Short: price below cloud, TK bearish, and downtrend (close < 1d EMA50)
            short_signal = (close_val < lower_cloud) and \
                           tk_bearish and \
                           (close_val < ema_50_1d_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below cloud OR TK turns bearish
            if close_val < upper_cloud or not tk_bullish:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above cloud OR TK turns bullish
            if close_val > lower_cloud or not tk_bearish:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_v1"
timeframe = "6h"
leverage = 1.0