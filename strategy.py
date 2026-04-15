#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily Donchian(20) channels
    donchian_high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_ratio_1d = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_ratio = volume / (volume_ratio_1d + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high_20_aligned[i]) or 
            np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price above daily EMA50 (bullish bias)
        # 2. Price breaks above daily Donchian(20) high with volume (bullish breakout)
        # 3. Volume confirmation: volume > 2.0x average
        if (close[i] > ema_50_1d_aligned[i] and
            close[i] > donchian_high_20_aligned[i] and
            volume_ratio[i] > 2.0):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below daily EMA50 (bearish bias)
        # 2. Price breaks below daily Donchian(20) low with volume (bearish breakdown)
        # 3. Volume confirmation: volume > 2.0x average
        elif (close[i] < ema_50_1d_aligned[i] and
              close[i] < donchian_low_20_aligned[i] and
              volume_ratio[i] > 2.0):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_DailyEMA50_Donchian20_Volume2x_Breakout_v1"
timeframe = "12h"
leverage = 1.0