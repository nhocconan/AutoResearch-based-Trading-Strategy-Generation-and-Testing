#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Donchian20_Breakout_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Daily Donchian Channels (20-day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-day rolling max/min (using previous day's data to avoid look-ahead)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Fill initial values
    donchian_high[:20] = high_1d[:20].max()
    donchian_low[:20] = low_1d[:20].min()
    
    # Align to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === Daily Trend Filter: 50-day EMA ===
    close_1d = df_1d['close'].values
    close_series_1d = pd.Series(close_1d)
    ema50 = close_series_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # === 6h Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        dh_val = donchian_high_aligned[i]
        dl_val = donchian_low_aligned[i]
        ema50_val = ema50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(dh_val) or np.isnan(dl_val) or 
            np.isnan(ema50_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian High with volume and uptrend filter
            if close_val > dh_val and vol_ratio_val > 1.5 and close_val > ema50_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian Low with volume and downtrend filter
            elif close_val < dl_val and vol_ratio_val > 1.5 and close_val < ema50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below Donchian Low OR trend turns bearish
            if close_val < dl_val or close_val < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above Donchian High OR trend turns bullish
            if close_val > dh_val or close_val > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals