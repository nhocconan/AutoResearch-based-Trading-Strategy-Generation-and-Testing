#!/usr/bin/env python3
"""
Hypothesis: 1-day Donchian(20) breakout with 1-week volatility filter and volume confirmation.
Long when price breaks above 20-day high and 1-week ATR ratio > 0.8 (not too choppy).
Short when price breaks below 20-day low and 1-week ATR ratio > 0.8.
Exit when price crosses 10-day EMA or ATR ratio < 0.6 (choppy market).
Uses daily timeframe for structure, weekly for regime filter to avoid false breakouts in chop.
Works in trending markets by capturing breakouts; avoids whipsaws in ranging markets via volatility filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Donchian channels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 20-day Donchian channels
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # 10-day EMA for exit
    ema_10 = pd.Series(df_1d['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Align 1D indicators to lower timeframe
    donchian_high = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_low = align_htf_to_ltf(prices, df_1d, low_20)
    ema_10_aligned = align_htf_to_ltf(prices, df_1d, ema_10)
    
    # Load 1-week data for ATR-based volatility filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # 14-period ATR for volatility measurement
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = np.inf  # First TR is invalid
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR as percentage of price for normalization
    atr_pct_1w = atr_1w / close_1w
    # Use 50-period average of ATR% to determine if volatility is normal
    avg_atr_pct_1w = pd.Series(atr_pct_1w).rolling(window=50, min_periods=50).mean().values
    atr_ratio_1w = atr_pct_1w / avg_atr_pct_1w  # >1 = more volatile than average
    
    # Align volatility filter
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio_1w)
    
    # Volume confirmation: 1-day volume vs 20-day average
    volume_1d = df_1d['volume'].values
    avg_volume_20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume_1d / avg_volume_20d  # >1 = above average volume
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_10_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(volume_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high with sufficient volatility and volume
            if (close[i] > donchian_high[i] and 
                atr_ratio_aligned[i] > 0.8 and  # Not too choppy
                volume_ratio_aligned[i] > 1.2):  # Above average volume
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with sufficient volatility and volume
            elif (close[i] < donchian_low[i] and 
                  atr_ratio_aligned[i] > 0.8 and 
                  volume_ratio_aligned[i] > 1.2):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below 10-day EMA OR volatility drops (choppy market)
                if close[i] < ema_10_aligned[i] or atr_ratio_aligned[i] < 0.6:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above 10-day EMA OR volatility drops
                if close[i] > ema_10_aligned[i] or atr_ratio_aligned[i] < 0.6:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wATR_Volume_Filter"
timeframe = "1d"
leverage = 1.0