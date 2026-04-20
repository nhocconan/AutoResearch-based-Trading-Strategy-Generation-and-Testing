#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Donchian_Breakout_VolumeTrend_v2"
timezone = "UTC"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # === Daily Donchian Channel (20-day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period high and low
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === 12h Trend Filter: 30-period EMA ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    close_series = pd.Series(close)
    ema30 = close_series.ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # === Volume Confirmation: 20-period average ===
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough data for indicators
        # Get values
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        ema30_val = ema30[i]
        d_high = donchian_high_aligned[i]
        d_low = donchian_low_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(ema30_val) or 
            np.isnan(d_high) or np.isnan(d_low)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high with volume and above EMA
            if (close_val > d_high and 
                vol_ratio_val > 1.8 and
                close_val > ema30_val):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with volume and below EMA
            elif (close_val < d_low and 
                  vol_ratio_val > 1.8 and
                  close_val < ema30_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below Donchian low or volume dries up
            if close_val < d_low or vol_ratio_val < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above Donchian high or volume dries up
            if close_val > d_high or vol_ratio_val < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals