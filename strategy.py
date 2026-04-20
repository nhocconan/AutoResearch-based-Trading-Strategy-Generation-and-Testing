#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_Ichimoku_TK_Cross_Cloud_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 26:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # === Weekly Ichimoku Components ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    
    # === Daily Trend Filter ===
    close_1d = df_1d['close'].values
    # 50-period EMA for daily trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 6h Price and Volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike filter: 2x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_spike = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        ema50 = ema50_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        # Skip if any value is NaN
        if (np.isnan(tenkan) or np.isnan(kijun) or np.isnan(span_a) or 
            np.isnan(span_b) or np.isnan(ema50) or np.isnan(vol_spike_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud top and bottom
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # Bullish TK cross: Tenkan crosses above Kijun
        tk_cross_bull = tenkan > kijun
        # Bearish TK cross: Tenkan crosses below Kijun
        tk_cross_bear = tenkan < kijun
        
        if position == 0:
            # Long: Bullish TK cross above cloud, price above cloud, daily uptrend, volume spike
            if (tk_cross_bull and 
                close_val > cloud_top and 
                close_val > ema50 and
                vol_spike_val > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: Bearish TK cross below cloud, price below cloud, daily downtrend, volume spike
            elif (tk_cross_bear and 
                  close_val < cloud_bottom and 
                  close_val < ema50 and
                  vol_spike_val > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price closes below cloud or TK cross turns bearish
            if close_val < cloud_bottom or (tenkan < kijun):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price closes above cloud or TK cross turns bullish
            if close_val > cloud_top or (tenkan > kijun):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals