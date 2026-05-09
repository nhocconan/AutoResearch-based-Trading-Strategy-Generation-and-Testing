#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_ChopFilter_DonchianBreakout"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend: EMA200 for long-term trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    ema200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily chop filter: Choppiness Index (14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_1d - low_1d
    
    # ATR(14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 days
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR(14)) / (HH(14) - LL(14))) / log10(14)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_ll = hh_1d - ll_1d
    chop = 100 * np.log10(tr_sum / hh_ll) / np.log10(14)
    chop = np.where(hh_ll == 0, 100, chop)  # avoid division by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12h Donchian channels (20)
    donchian_len = 20
    highest_high = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # Volume filter: volume > 1.5x 20-period SMA
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(chop_aligned[i]) or \
           np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper, weekly uptrend, not choppy, volume
            if (price > highest_high[i] and 
                price > ema200_1w_aligned[i] and 
                chop_aligned[i] < 61.8 and  # not choppy (trending)
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price breaks below Donchian lower, weekly downtrend, not choppy, volume
            elif (price < lowest_low[i] and 
                  price < ema200_1w_aligned[i] and 
                  chop_aligned[i] < 61.8 and  # not choppy (trending)
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price breaks below Donchian lower or chop increases
            if (price < lowest_low[i] or 
                chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian upper or chop increases
            if (price > highest_high[i] or 
                chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals