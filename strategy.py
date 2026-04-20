#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian20_Breakout_VolumeTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Daily Donchian Channels (20-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels on daily data
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe (previous day's channels)
    donchian_high_shifted = np.roll(donchian_high, 1)
    donchian_low_shifted = np.roll(donchian_low, 1)
    donchian_high_shifted[0] = donchian_high[0]
    donchian_low_shifted[0] = donchian_low[0]
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_shifted)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_shifted)
    
    # === Volume Trend Filter ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === Price Trend Filter: 4h EMA50 > EMA200 for long, < for short ===
    close_series = pd.Series(prices['close'].values)
    ema50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        upper_dc = donchian_high_aligned[i]
        lower_dc = donchian_low_aligned[i]
        ema50_val = ema50[i]
        ema200_val = ema200[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(upper_dc) or 
            np.isnan(lower_dc) or np.isnan(ema50_val) or np.isnan(ema200_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper Donchian with volume confirmation and uptrend
            if close_val > upper_dc and vol_ratio_val > 2.0 and ema50_val > ema200_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian with volume confirmation and downtrend
            elif close_val < lower_dc and vol_ratio_val > 2.0 and ema50_val < ema200_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below lower Donchian OR trend breaks down
            if close_val < lower_dc or ema50_val < ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above upper Donchian OR trend breaks up
            if close_val > upper_dc or ema50_val > ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals