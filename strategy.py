#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(21) for trend filter
    ema_21_1d = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Calculate 1d Donchian(20) channels
    donchian_high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Calculate 1d volume MA(20) for volume filter
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1d_aligned[i]) or 
            np.isnan(donchian_high_20_aligned[i]) or 
            np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x daily average volume (scaled)
        # Approximate: daily volume / 4 = expected 6h volume (since 24h/6h = 4)
        vol_threshold = vol_ma_20_aligned[i] / 4.0
        volume_confirm = volume[i] > vol_threshold
        
        # Long conditions:
        # 1. Price above daily EMA21 (bullish bias)
        # 2. Price breaks above daily Donchian(20) high with volume confirmation
        if (close[i] > ema_21_1d_aligned[i] and
            close[i] > donchian_high_20_aligned[i] and
            volume_confirm):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below daily EMA21 (bearish bias)
        # 2. Price breaks below daily Donchian(20) low with volume confirmation
        elif (close[i] < ema_21_1d_aligned[i] and
              close[i] < donchian_low_20_aligned[i] and
              volume_confirm):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_DailyEMA21_Donchian20_Volume_Breakout_v1"
timeframe = "6h"
leverage = 1.0