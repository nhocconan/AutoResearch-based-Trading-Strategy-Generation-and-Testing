#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Donchian20_Volume_Spike_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # === 12h Donchian Channel (20-period) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian upper/lower bands
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # === 4h: Volume Spike and Trend Filter ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike detection: volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma20)
    
    # 4h EMA50 trend filter
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        vol_spike_val = vol_spike[i]
        ema50_val = ema50[i]
        upper_val = donchian_upper_aligned[i]
        lower_val = donchian_lower_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_spike_val) or np.isnan(ema50_val) or 
            np.isnan(upper_val) or np.isnan(lower_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper with volume spike and uptrend
            if (close_val > upper_val and 
                vol_spike_val and 
                close_val > ema50_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with volume spike and downtrend
            elif (close_val < lower_val and 
                  vol_spike_val and 
                  close_val < ema50_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below Donchian lower or volume drops
            if close_val < lower_val or not vol_spike_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above Donchian upper or volume drops
            if close_val > upper_val or not vol_spike_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals