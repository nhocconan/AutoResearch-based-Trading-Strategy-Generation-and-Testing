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
    
    # Get weekly data for 12h trading
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly ATR for volatility filtering
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range and ATR
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_period = 14
    atr_1w = np.full(len(close_1w), np.nan)
    for i in range(atr_period, len(tr)):
        atr_1w[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Align weekly ATR to 12h timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate 12-period Donchian channels on 12h data
    lookback = 12
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # 12-period average volume for spike detection
    vol_period = 12
    vol_ma = np.full(n, np.nan)
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period: need at least 12 for Donchian, 12 for volume, 14 for ATR
    start_idx = max(lookback, vol_period, atr_period)
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(atr_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volatility filter: only trade when volatility is above average
        vol_filter = atr_1w_aligned[i] > np.nanmean(atr_1w_aligned[max(0, i-50):i]) if not np.isnan(np.nanmean(atr_1w_aligned[max(0, i-50):i])) else False
        
        if position == 0:
            # Long breakout: price breaks above Donchian high with volume and volatility
            if price > highest_high[i] and vol_ratio > 1.5 and vol_filter:
                signals[i] = size
                position = 1
            # Short breakdown: price breaks below Donchian low with volume and volatility
            elif price < lowest_low[i] and vol_ratio > 1.5 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low
            if price < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above Donchian high
            if price > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian12_WeeklyATR_Volume"
timeframe = "12h"
leverage = 1.0