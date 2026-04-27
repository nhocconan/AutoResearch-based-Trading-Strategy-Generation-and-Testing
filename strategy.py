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
    
    # Get 1d data for calculations (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day Exponential Moving Average (34-period) for trend
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        multiplier = 2 / (34 + 1)
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * multiplier) + (ema_34_1d[i-1] * (1 - multiplier))
    
    # Calculate 1-day Bollinger Bands (20, 2.0)
    sma_20_1d = np.full(len(close_1d), np.nan)
    std_20_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 20:
        for i in range(19, len(close_1d)):
            sma_20_1d[i] = np.mean(close_1d[i-19:i+1])
            std_20_1d[i] = np.std(close_1d[i-19:i+1])
    
    upper_bb_1d = sma_20_1d + (2 * std_20_1d)
    lower_bb_1d = sma_20_1d - (2 * std_20_1d)
    
    # Align 1d indicators to daily timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    upper_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    
    # Calculate 10-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 10
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(34, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(upper_bb_1d_aligned[i]) or 
            np.isnan(lower_bb_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 2.0x average volume
        vol_filter = vol_ratio > 2.0
        
        if position == 0:
            # Long: Price above EMA34 and breaks above upper Bollinger Band with volume
            if price > ema_34_1d_aligned[i] and price > upper_bb_1d_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: Price below EMA34 and breaks below lower Bollinger Band with volume
            elif price < ema_34_1d_aligned[i] and price < lower_bb_1d_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below EMA34 or volatility spike (potential reversal)
            if price < ema_34_1d_aligned[i] or (vol_ratio > 3.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price crosses above EMA34 or volatility spike (potential reversal)
            if price > ema_34_1d_aligned[i] or (vol_ratio > 3.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_EMA34_BB20_Volume"
timeframe = "1d"
leverage = 1.0