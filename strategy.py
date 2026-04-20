#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian20_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily Donchian Channel (20-day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period high and low
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === 4h Indicators ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 50-period EMA for trend filter
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume ratio (20-period average)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        ema50_val = ema50[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(ema50_val) or 
            np.isnan(upper) or np.isnan(lower)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper Donchian with volume and trend
            if (close_val > upper and 
                vol_ratio_val > 1.8 and
                close_val > ema50_val):
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian with volume and counter-trend
            elif (close_val < lower and 
                  vol_ratio_val > 1.8 and
                  close_val < ema50_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below lower Donchian or trend breaks
            if close_val < lower or close_val < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above upper Donchian or trend breaks
            if close_val > upper or close_val > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals