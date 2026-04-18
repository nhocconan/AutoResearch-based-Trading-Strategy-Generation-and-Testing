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
    
    # Get 1D data for daily ATR and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR (14-period)
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        atr_1d[i] = np.mean(tr_1d[i-13:i+1])
    
    # Calculate daily Bollinger Bands (20-period, 2.0 std)
    sma_20 = np.full(len(close_1d), np.nan)
    std_20 = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        sma_20[i] = np.mean(close_1d[i-20:i])
        std_20[i] = np.std(close_1d[i-20:i])
    upper_bb = sma_20 + 2.0 * std_20
    lower_bb = sma_20 - 2.0 * std_20
    
    # Align daily indicators to 6h timeframe
    atr_1d_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    upper_bb_6h = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_6h = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Calculate 6h Bollinger Bands (20-period, 2.0 std) for mean reversion signals
    sma_20_6h = np.full(n, np.nan)
    std_20_6h = np.full(n, np.nan)
    for i in range(20, n):
        sma_20_6h[i] = np.mean(close[i-20:i])
        std_20_6h[i] = np.std(close[i-20:i])
    upper_bb_6h_local = sma_20_6h + 2.0 * std_20_6h
    lower_bb_6h_local = sma_20_6h - 2.0 * std_20_6h
    
    # Calculate volume average (20-period) for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # need Bollinger Bands and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_1d_6h[i]) or np.isnan(upper_bb_6h[i]) or np.isnan(lower_bb_6h[i]) or
            np.isnan(vol_ma[i]) or np.isnan(sma_20_6h[i]) or np.isnan(std_20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when daily ATR > 0.5 * 6h price (avoid extremely low volatility)
        vol_filter = atr_1d_6h[i] > 0.5 * (high[i] - low[i])
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long entry: price touches lower Bollinger Band with volume confirmation and volatility filter
            if close[i] <= lower_bb_6h_local[i] and vol_confirmed and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price touches upper Bollinger Band with volume confirmation and volatility filter
            elif close[i] >= upper_bb_6h_local[i] and vol_confirmed and vol_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses above the 20-period SMA (mean reversion target)
            if close[i] > sma_20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses below the 20-period SMA (mean reversion target)
            if close[i] < sma_20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Bollinger_MeanReversion_VolumeVolatilityFilter"
timeframe = "6h"
leverage = 1.0