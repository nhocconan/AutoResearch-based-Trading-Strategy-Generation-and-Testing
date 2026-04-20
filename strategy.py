#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Donchian20_Volume_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA Trend Filter (21)
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Donchian Channel (20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Rolling window for Donchian high/low
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Daily EMA Trend Filter (21)
    close_series = pd.Series(close_1d)
    ema21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume ratio (20-period)
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Get values
        close_val = close_1d[i]
        vol_ratio_val = vol_ratio[i]
        ema21_val = ema21[i]
        ema21_1w_val = ema21_1w_aligned[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(ema21_val) or np.isnan(ema21_1w_val) or 
            np.isnan(donchian_high_val) or np.isnan(donchian_low_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume confirmation and uptrend (daily & weekly)
            if (close_val > donchian_high_val and 
                vol_ratio_val > 1.5 and 
                close_val > ema21_val and
                close_val > ema21_1w_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume confirmation and downtrend (daily & weekly)
            elif (close_val < donchian_low_val and 
                  vol_ratio_val > 1.5 and 
                  close_val < ema21_val and
                  close_val < ema21_1w_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below Donchian low or volume dries up
            if close_val < donchian_low_val or vol_ratio_val < 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above Donchian high or volume dries up
            if close_val > donchian_high_val or vol_ratio_val < 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals