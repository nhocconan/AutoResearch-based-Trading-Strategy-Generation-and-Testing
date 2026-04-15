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
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA(21) for weekly trend
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate 1w Williams %R(14) for overbought/oversold
    highest_high_14 = pd.Series(df_1w['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1w['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r_1w = -100 * (highest_high_14 - df_1w['close'].values) / (highest_high_14 - lowest_low_14 + 1e-10)
    williams_r_1w_aligned = align_htf_to_ltf(prices, df_1w, williams_r_1w)
    
    # Calculate 6h Donchian(20) channels for breakout
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(williams_r_1w_aligned[i]) or 
            np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above/below weekly EMA21
        weekly_uptrend = close[i] > ema_21_1w_aligned[i]
        weekly_downtrend = close[i] < ema_21_1w_aligned[i]
        
        # Williams %R conditions: oversold < -80, overbought > -20
        oversold = williams_r_1w_aligned[i] < -80
        overbought = williams_r_1w_aligned[i] > -20
        
        # Long conditions:
        # 1. Weekly uptrend (price above weekly EMA21)
        # 2. Weekly Williams %R oversold (< -80) - buying opportunity in uptrend
        # 3. Price breaks above 6h Donchian(20) high with volume confirmation
        # 4. Volume > 1.5x average
        if (weekly_uptrend and oversold and
            close[i] > donchian_high_20[i] and
            volume_ratio[i] > 1.5):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Weekly downtrend (price below weekly EMA21)
        # 2. Weekly Williams %R overbought (> -20) - selling opportunity in downtrend
        # 3. Price breaks below 6h Donchian(20) low with volume confirmation
        # 4. Volume > 1.5x average
        elif (weekly_downtrend and overbought and
              close[i] < donchian_low_20[i] and
              volume_ratio[i] > 1.5):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyEMA21_WilliamsR_Donchian20_Breakout_v1"
timeframe = "6h"
leverage = 1.0