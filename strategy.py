#!/usr/bin/env python3
# 4h_donchian_volume_regime_v1
# Hypothesis: Donchian channel breakout with volume confirmation and choppiness regime filter on 4h timeframe.
# Uses 1d trend filter to avoid counter-trend trades. Designed to work in both bull and bear markets by
# filtering for trending regimes (low choppiness). Target: 20-50 trades/year (80-200 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1h Donchian channels (20-period)
    donch_len = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(donch_len - 1, n):
        upper[i] = np.max(high[i-donch_len+1:i+1])
        lower[i] = np.min(low[i-donch_len+1:i+1])
    
    # Calculate 1d trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = np.zeros(len(df_1d))
    ema50_1d[0] = close_1d[0]
    alpha = 2 / (50 + 1)
    for i in range(1, len(df_1d)):
        ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # 1d trend: 1 if close > EMA50, -1 if close < EMA50
    trend_1d = np.where(close_1d > ema50_1d, 1, -1)
    trend_4h = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate choppiness index (14-period) on 4h
    chop_len = 14
    atr = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(chop_len, n):
        atr[i] = np.mean(tr[i-chop_len+1:i+1])
    
    chop = np.full(n, np.nan)
    for i in range(chop_len, n):
        if atr[i] > 0:
            sum_tr = np.sum(tr[i-chop_len+1:i+1])
            max_range = np.max(high[i-chop_len+1:i+1]) - np.min(low[i-chop_len+1:i+1])
            if max_range > 0:
                chop[i] = 100 * np.log10(sum_tr / max_range) / np.log10(chop_len)
    
    # Volume filter: 20-period average
    vol_ma = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(trend_4h[i]) or np.isnan(chop[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma[i] * 1.5
        
        # Choppiness filter: only trade when trending (CHOP < 38.2)
        trending = chop[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: close below lower Donchian or trend turns bearish or choppy
            if close[i] < lower[i] or trend_4h[i] == -1 or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: close above upper Donchian or trend turns bullish or choppy
            if close[i] > upper[i] or trend_4h[i] == 1 or not trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: break above upper Donchian with volume, bullish trend, and trending market
            if (close[i] > upper[i] and 
                vol_ok and 
                trend_4h[i] == 1 and 
                trending):
                position = 1
                signals[i] = 0.25
            # Enter short: break below lower Donchian with volume, bearish trend, and trending market
            elif (close[i] < lower[i] and 
                  vol_ok and 
                  trend_4h[i] == -1 and 
                  trending):
                position = -1
                signals[i] = -0.25
    
    return signals