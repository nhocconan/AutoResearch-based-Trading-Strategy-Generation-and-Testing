#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly linear regression slope (5-period) for long-term trend
    close_1w = df_1w['close'].values
    x = np.arange(5)
    slope_1w = np.zeros_like(close_1w, dtype=float)
    for i in range(4, len(close_1w)):
        y = close_1w[i-4:i+1]
        if np.any(np.isnan(y)):
            slope_1w[i] = np.nan
        else:
            slope = np.polyfit(x, y, 1)[0]
            slope_1w[i] = slope
    slope_1w_aligned = align_htf_to_ltf(prices, df_1w, slope_1w)
    
    # Calculate daily Donchian channels (20) for breakout signals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Calculate daily ATR (14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        upper_val = donchian_upper_aligned[i]
        lower_val = donchian_lower_aligned[i]
        slope_val = slope_1w_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_val) or np.isnan(lower_val) or 
            np.isnan(slope_val) or np.isnan(atr_1d_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band with upward weekly trend
            if close_val > upper_val and slope_val > 0 and atr_1d_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band with downward weekly trend
            elif close_val < lower_val and slope_val < 0 and atr_1d_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below lower Donchian band
            if close_val < lower_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above upper Donchian band
            if close_val > upper_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 1d_1wTrend_1dDonchian_NoATRStop
# Uses 1-week linear regression slope for long-term trend direction
# Enters long when 1d price breaks above 1d upper Donchian band with upward weekly trend
# Enters short when 1d price breaks below 1d lower Donchian band with downward weekly trend
# Exits on opposite band touch (no ATR stop to reduce whipsaw)
# Designed for 1d timeframe with ~10-30 trades/year
name = "1d_1wTrend_1dDonchian_NoATRStop"
timeframe = "1d"
leverage = 1.0