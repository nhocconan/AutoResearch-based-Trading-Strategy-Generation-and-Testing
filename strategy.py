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
    
    # Get weekly data for ATR filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR for regime filter
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1w[1:] - close_1w[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr_1w = pd.Series(tr1).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_1w = np.concatenate([np.full(14, np.nan), atr_1w[14:]])
    
    # Align weekly ATR to daily timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate daily ATR for stop loss
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr = np.concatenate([np.full(14, np.nan), atr[14:]])
    
    # Donchian channel (20-day)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 34  # Need ATR (14*2), Donchian (20), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_1w_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Weekly ATR filter: only trade when volatility is elevated (above median)
        # Calculate median of ATR_1w using expanding window
        if i >= 50:  # Need sufficient history for median
            atr_median = np.nanmedian(atr_1w_aligned[:i+1])
            vol_filter = atr_1w_aligned[i] > atr_median
        else:
            vol_filter = True  # Not enough data, allow trading initially
        
        # Volume filter
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and volatility filter
            if (close[i] > highest_high[i] and volume_filter and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and volatility filter
            elif (close[i] < lowest_low[i] and volume_filter and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Trail stop: exit if price drops below highest high - 2*ATR
            if close[i] < (highest_high[i] - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Trail stop: exit if price rises above lowest low + 2*ATR
            if close[i] > (lowest_low[i] + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian_Breakout_Volume_VolatilityFilter"
timeframe = "1d"
leverage = 1.0