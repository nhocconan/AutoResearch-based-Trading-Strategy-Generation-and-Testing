#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian20_VolumeSpike_Trend_v2"
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
    
    # === Daily Donchian Channels (20-day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: highest high of last 20 days (including current)
    upper_band = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 days (including current)
    lower_band = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe (shifted to avoid look-ahead)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # === 4h Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === 4h Trend Filter: EMA21 vs EMA50 ===
    close = prices['close'].values
    ema21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Get values
        close_val = prices['close'].iloc[i]
        upper_val = upper_band_aligned[i]
        lower_val = lower_band_aligned[i]
        vol_ratio_val = vol_ratio[i]
        ema21_val = ema21[i]
        ema50_val = ema50[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_val) or np.isnan(lower_val) or 
            np.isnan(vol_ratio_val) or np.isnan(ema21_val) or np.isnan(ema50_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper Donchian band with volume confirmation and uptrend
            if close_val > upper_val and vol_ratio_val > 2.0 and ema21_val > ema50_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian band with volume confirmation and downtrend
            elif close_val < lower_val and vol_ratio_val > 2.0 and ema21_val < ema50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below lower Donchian band OR trend reverses
            if close_val < lower_val or ema21_val < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above upper Donchian band OR trend reverses
            if close_val > upper_val or ema21_val > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals