#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 20-period High/Low on 1d for Donchian channel
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper/lower bands
    donch_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 1d timeframe
    donch_upper_aligned = align_htf_to_ltf(prices, df_1d, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1d, donch_lower)
    
    # Calculate 1d ATR for volatility filter
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 50-period SMA on 1w close for trend
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = prices['close'].iloc[i]
        upper_val = donch_upper_aligned[i]
        lower_val = donch_lower_aligned[i]
        atr_val = atr_1d_aligned[i]
        sma_val = sma_50_1w_aligned[i]
        
        # Skip if any value is NaN
        if np.isnan(upper_val) or np.isnan(lower_val) or np.isnan(atr_val) or np.isnan(sma_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper + above weekly SMA + volatility filter
            if close_val > upper_val and close_val > sma_val and atr_val > 0.01 * close_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + below weekly SMA + volatility filter
            elif close_val < lower_val and close_val < sma_val and atr_val > 0.01 * close_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian lower
            if close_val < lower_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian upper
            if close_val > upper_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 1d_Donchian20_WeeklySMA50_VolatilityFilter_v1
# Uses daily Donchian(20) breakout for entry
# Requires price to be above/below weekly 50 SMA for trend filter
# Volatility filter: 1d ATR > 1% of price to avoid low-vol chop
# Exit when price reverses to opposite Donchian band
# Designed for 1d timeframe with ~10-25 trades/year
name = "1d_Donchian20_WeeklySMA50_VolatilityFilter_v1"
timeframe = "1d"
leverage = 1.0