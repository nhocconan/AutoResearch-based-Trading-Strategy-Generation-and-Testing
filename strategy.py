#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Donchian20_VolumeSpike_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # === 12h Donchian Channel (20-period) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate rolling max/min with proper alignment
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe (waits for 12h bar close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # === Volume Spike Filter (4h) ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20>0)
    
    # === 4h EMA Trend Filter ===
    close_series = pd.Series(prices['close'].values)
    ema50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema50[i]) or np.isnan(ema200[i]) or vol_ma20[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = prices['close'].iloc[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        vol_ratio_val = vol_ratio[i]
        ema50_val = ema50[i]
        ema200_val = ema200[i]
        
        if position == 0:
            # Long: Break above Donchian high with volume spike and uptrend
            if close_val > donchian_high_val and vol_ratio_val > 2.0 and ema50_val > ema200_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with volume spike and downtrend
            elif close_val < donchian_low_val and vol_ratio_val > 2.0 and ema50_val < ema200_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below Donchian low OR trend breaks down
            if close_val < donchian_low_val or ema50_val < ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above Donchian high OR trend breaks up
            if close_val > donchian_high_val or ema50_val > ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals