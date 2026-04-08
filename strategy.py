#!/usr/bin/env python3
# 1h_4h_1d_camarilla_pivot_volume_regime_v1
# Hypothesis: Camarilla pivot levels on 4h with volume spike and 1d regime filter on 1h timeframe.
# Long: Price > H4 (camarilla resistance) AND volume > 1.5 * volume_ma(20) AND close > 1d EMA50
# Short: Price < L4 (camarilla support) AND volume > 1.5 * volume_ma(20) AND close < 1d EMA50
# Exit: Opposite signal or price reverts to Pivot Point (PP)
# Uses 1h primary timeframe with 4h for Camarilla pivot calculation and 1d for EMA50 regime filter.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag and avoid overtrading.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_pivot_volume_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar
    # Typical Price = (High + Low + Close) / 3
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    # Range = High - Low
    range_4h = high_4h - low_4h
    
    # Camarilla levels:
    # H4 = Close + 1.1 * Range / 2
    # L4 = Close - 1.1 * Range / 2
    # PP = (High + Low + Close) / 3 (same as typical price)
    h4_4h = close_4h + 1.1 * range_4h / 2.0
    l4_4h = close_4h - 1.1 * range_4h / 2.0
    pp_4h = typical_price_4h  # (High + Low + Close) / 3
    
    # Align 4h Camarilla levels to 1h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_4h, h4_4h)
    l4_aligned = align_htf_to_ltf(prices, df_4h, l4_4h)
    pp_aligned = align_htf_to_ltf(prices, df_4h, pp_4h)
    
    # Get 1d data for regime filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA50 on 1d with min_periods
    close_1d_s = pd.Series(close_1d)
    ema50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 1h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5 * volume_ma(20)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after volume_ma warmup
        # Skip if any required data is NaN
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Apply session filter
        if not session_filter[i]:
            # Outside session: flatten position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Opposite signal (price < L4)
            # 2. Price reverts to Pivot Point (PP)
            if close[i] < l4_aligned[i] or abs(close[i] - pp_aligned[i]) < 0.001 * pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Opposite signal (price > H4)
            # 2. Price reverts to Pivot Point (PP)
            if close[i] > h4_aligned[i] or abs(close[i] - pp_aligned[i]) < 0.001 * pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Long entry: Price > H4 AND volume confirmation AND close > 1d EMA50 (bull regime)
            if close[i] > h4_aligned[i] and volume_confirm[i] and close[i] > ema50_1d_aligned[i]:
                position = 1
                signals[i] = 0.20
            # Short entry: Price < L4 AND volume confirmation AND close < 1d EMA50 (bear regime)
            elif close[i] < l4_aligned[i] and volume_confirm[i] and close[i] < ema50_1d_aligned[i]:
                position = -1
                signals[i] = -0.20
    
    return signals