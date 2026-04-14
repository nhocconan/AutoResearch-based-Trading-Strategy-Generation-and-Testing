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
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily SMA(21) for trend
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    sma_21_1d = close_1d_series.rolling(window=21, min_periods=21).mean().values
    
    # Calculate median volume for volume spike filter
    vol_median = np.nanmedian(volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(100, n):
        # Get aligned daily SMA
        sma_21_1d_i = align_htf_to_ltf(prices, df_1d, sma_21_1d)[i]
        
        if np.isnan(sma_21_1d_i):
            continue
        
        # Volume spike filter
        volume_spike = volume[i] > 1.3 * vol_median
        
        # Long conditions:
        # 1. Price above daily SMA21 (uptrend)
        # 2. Volume spike
        if position == 0 and volume_spike:
            if close[i] > sma_21_1d_i:
                position = 1
                signals[i] = position_size
            # Short conditions:
            # 1. Price below daily SMA21 (downtrend)
            elif close[i] < sma_21_1d_i:
                position = -1
                signals[i] = -position_size
        
        # Exit conditions: price crosses back across daily SMA21
        elif position == 1:
            if close[i] < sma_21_1d_i:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            if close[i] > sma_21_1d_i:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_DailySMA21_Volume_Spike_Filter"
timeframe = "6h"
leverage = 1.0