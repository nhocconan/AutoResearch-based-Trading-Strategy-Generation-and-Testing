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
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(21) for trend filter
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate weekly Donchian(10) channels
    donchian_high_10 = pd.Series(df_1w['high'].values).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(df_1w['low'].values).rolling(window=10, min_periods=10).min().values
    donchian_high_10_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_10)
    donchian_low_10_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_10)
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(donchian_high_10_aligned[i]) or 
            np.isnan(donchian_low_10_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to weekly EMA21
        trend_filter = close[i] > ema_21_1w_aligned[i]
        
        # Long conditions:
        # 1. Price above weekly EMA21 (bullish bias)
        # 2. Price breaks above weekly Donchian(10) high with volume (bullish breakout)
        # 3. Volume confirmation: volume > 1.5x average
        if (trend_filter and
            close[i] > donchian_high_10_aligned[i] and
            volume_ratio[i] > 1.5):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below weekly EMA21 (bearish bias)
        # 2. Price breaks below weekly Donchian(10) low with volume (bearish breakdown)
        # 3. Volume confirmation: volume > 1.5x average
        elif (not trend_filter and
              close[i] < donchian_low_10_aligned[i] and
              volume_ratio[i] > 1.5):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Vol_Regime_WDonchian10_1wEMA21_Breakout_v1"
timeframe = "12h"
leverage = 1.0