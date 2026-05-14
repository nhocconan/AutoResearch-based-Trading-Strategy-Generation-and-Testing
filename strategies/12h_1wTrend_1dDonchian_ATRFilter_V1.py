#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1w and 1d data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1w linear regression slope (5-period) for long-term trend
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
    
    # Calculate 1d Donchian channels (20) for breakout signals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Calculate 1d ATR (14) for volatility filter and stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 12h ATR (14) for stoploss
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        upper_val = donchian_upper_aligned[i]
        lower_val = donchian_lower_aligned[i]
        slope_val = slope_1w_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        atr_12h_val = atr_12h[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_val) or np.isnan(lower_val) or 
            np.isnan(slope_val) or np.isnan(atr_1d_val) or 
            np.isnan(atr_12h_val)):
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
            # Long exit: price closes below lower Donchian band or ATR-based stop
            if close_val < lower_val or close_val < prices['high'].iloc[i] - 2.0 * atr_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above upper Donchian band or ATR-based stop
            if close_val > upper_val or close_val > prices['low'].iloc[i] + 2.0 * atr_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 12h_1wTrend_1dDonchian_ATRFilter_V1
# Uses 1-week linear regression slope for long-term trend direction
# Enters long when 12h price breaks above 1d upper Donchian band with upward weekly trend
# Enters short when 12h price breaks below 1d lower Donchian band with downward weekly trend
# Uses 1d ATR as volatility filter to avoid choppy markets
# Exits on opposite band touch or 2*ATR stoploss (using 12h ATR)
# Designed for 12h timeframe with ~12-37 trades/year
name = "12h_1wTrend_1dDonchian_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0