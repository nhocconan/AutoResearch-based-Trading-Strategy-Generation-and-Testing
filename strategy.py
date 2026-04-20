#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian20_VolumeSpike_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily Donchian Channel (20-day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 20-period high and low (using previous day's data to avoid look-ahead)
    high_max = np.full_like(high_1d, np.nan)
    low_min = np.full_like(low_1d, np.nan)
    
    for i in range(len(high_1d)):
        if i >= 20:
            high_max[i] = np.max(high_1d[i-20:i])  # previous 20 days, not including current
            low_min[i] = np.min(low_1d[i-20:i])
        elif i > 0:
            high_max[i] = high_max[i-1]
            low_min[i] = low_min[i-1]
        else:
            high_max[i] = high_1d[0]
            low_min[i] = low_1d[0]
    
    # Align to 4h timeframe
    upper_band = align_htf_to_ltf(prices, df_1d, high_max)
    lower_band = align_htf_to_ltf(prices, df_1d, low_min)
    
    # === 4h Volume Spike Detection ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_std20 = vol_series.rolling(window=20, min_periods=20).std().values
    vol_zscore = np.where(vol_std20 > 0, (volume - vol_ma20) / vol_std20, 0)
    
    # === 4h Trend Filter: EMA(50) > EMA(200) for long, < for short ===
    close_series = pd.Series(prices['close'].values)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = prices['close'].iloc[i]
        upper_val = upper_band[i]
        lower_val = lower_band[i]
        vol_z = vol_zscore[i]
        ema50_val = ema_50[i]
        ema200_val = ema_200[i]
        
        # Skip if any value is NaN
        if (np.isnan(close_val) or np.isnan(upper_val) or np.isnan(lower_val) or
            np.isnan(vol_z) or np.isnan(ema50_val) or np.isnan(ema200_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper Donchian band with volume spike in bullish trend
            if close_val > upper_val and vol_z > 2.0 and ema50_val > ema200_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian band with volume spike in bearish trend
            elif close_val < lower_val and vol_z > 2.0 and ema50_val < ema200_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below lower band OR trend turns bearish
            if close_val < lower_val or ema50_val < ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above upper band OR trend turns bullish
            if close_val > upper_val or ema50_val > ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals