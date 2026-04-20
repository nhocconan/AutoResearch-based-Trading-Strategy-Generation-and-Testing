#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Choppiness_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for choppiness regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 14-period Choppiness Index on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for chop calculation
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Sum of TR over 14 periods
    tr_sum = np.full_like(tr, np.nan)
    for i in range(13, len(tr)):
        tr_sum[i] = np.nansum(tr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    hh = np.full_like(high_1d, np.nan)
    ll = np.full_like(low_1d, np.nan)
    for i in range(13, len(high_1d)):
        hh[i] = np.nanmax(high_1d[i-13:i+1])
        ll[i] = np.nanmin(low_1d[i-13:i+1])
    
    # Choppiness Index: 100 * log10(sum(tr) / (hh - ll)) / log10(14)
    chop = np.full_like(tr, np.nan)
    for i in range(13, len(tr)):
        if tr_sum[i] > 0 and hh[i] > ll[i]:
            chop[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(14)
    
    # Align chop to 4h timeframe (chop > 61.8 = ranging market)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate ATR for stop loss and Donchian channels (20-period on 4h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR (20-period)
    atr = np.full_like(tr, np.nan)
    for i in range(19, len(tr)):
        atr[i] = np.nanmean(tr[i-19:i+1])
    
    # Donchian Channels (20-period)
    donch_high = np.full_like(high, np.nan)
    donch_low = np.full_like(low, np.nan)
    for i in range(19, len(high)):
        donch_high[i] = np.nanmax(high[i-19:i+1])
        donch_low[i] = np.nanmin(low[i-19:i+1])
    
    # Volume average (20-period)
    vol = prices['volume'].values
    vol_avg = np.full_like(vol, np.nan)
    for i in range(19, len(vol)):
        vol_avg[i] = np.nanmean(vol[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Only trade in ranging markets (chop > 61.8)
        if np.isnan(chop_aligned[i]) or chop_aligned[i] <= 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        current_close = close[i]
        current_volume = vol[i]
        current_atr = atr[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume confirmation
            if (not np.isnan(donch_high[i]) and 
                current_close > donch_high[i] and 
                not np.isnan(vol_avg[i]) and 
                current_volume > 1.5 * vol_avg[i]):
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            # Short: price breaks below Donchian low with volume confirmation
            elif (not np.isnan(donch_low[i]) and 
                  current_close < donch_low[i] and 
                  not np.isnan(vol_avg[i]) and 
                  current_volume > 1.5 * vol_avg[i]):
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or ATR stop loss
            if (not np.isnan(donch_low[i]) and 
                current_close < donch_low[i]):
                signals[i] = 0.0
                position = 0
            elif current_atr > 0 and current_close < entry_price - 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or ATR stop loss
            if (not np.isnan(donch_high[i]) and 
                current_close > donch_high[i]):
                signals[i] = 0.0
                position = 0
            elif current_atr > 0 and current_close > entry_price + 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals