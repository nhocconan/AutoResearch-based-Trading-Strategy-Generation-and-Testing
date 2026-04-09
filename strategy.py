#!/usr/bin/env python3
# 12h_camarilla_1d_trend_volume_v1
# Hypothesis: 12h strategy using 1d Camarilla pivot levels for entry/exit, volume confirmation, and 1w ADX regime filter.
# Long: price > H3 + volume spike + ADX > 25 (trending)
# Short: price < L3 + volume spike + ADX > 25 (trending)
# Exit: price crosses H4/L4 levels OR ADX < 20 (range)
# Uses discrete sizing (±0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Pivot = (high + low + close) / 3
    pivot = (pd.Series(high_1d).shift(1) + pd.Series(low_1d).shift(1) + pd.Series(close_1d).shift(1)) / 3
    # Range = high - low
    rang = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    
    # Camarilla levels
    h3 = pivot + rang * 1.1 / 4
    l3 = pivot - rang * 1.1 / 4
    h4 = pivot + rang * 1.1 / 2
    l4 = pivot - rang * 1.1 / 2
    
    # 1w HTF data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14-period) on 1w
    plus_dm = pd.Series(high_1w).diff()
    minus_dm = pd.Series(low_1w).diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = pd.Series(high_1w).diff().abs()
    tr2 = pd.Series(low_1w).diff().abs()
    tr3 = pd.Series(close_1w).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    # Align all indicators to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3.values)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4.values)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4.values)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx.values)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below H4 OR ADX < 20 (range)
            if close[i] < h4_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above L4 OR ADX < 20 (range)
            if close[i] > l4_aligned[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and trending regime (ADX > 25)
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            trending = adx_aligned[i] > 25
            
            if volume_confirmed and trending:
                # Long: price > H3
                if close[i] > h3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price < L3
                elif close[i] < l3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals