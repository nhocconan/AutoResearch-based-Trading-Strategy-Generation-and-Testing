#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for 1D ATR and close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-day ATR (daily)
    if len(high_1d) < 15:
        return np.zeros(n)
    
    tr = np.zeros_like(high_1d)
    for i in range(1, len(high_1d)):
        tr[i] = max(high_1d[i] - low_1d[i],
                   abs(high_1d[i] - high_1d[i-1]),
                   abs(low_1d[i] - low_1d[i-1]))
    
    atr_1d = np.full_like(high_1d, np.nan)
    if len(high_1d) >= 15:
        atr_1d[14] = np.mean(tr[1:15])
        for i in range(15, len(high_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align daily ATR to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 10-period SMA of daily close (trend filter)
    if len(close_1d) < 10:
        return np.zeros(n)
    
    sma10_1d = np.full_like(close_1d, np.nan)
    for i in range(9, len(close_1d)):
        sma10_1d[i] = np.mean(close_1d[i-9:i+1])
    
    # Align daily SMA to 12h timeframe
    sma10_1d_aligned = align_htf_to_ltf(prices, df_1d, sma10_1d)
    
    # Calculate 20-period standard deviation of daily close (volatility filter)
    if len(close_1d) < 20:
        return np.zeros(n)
    
    std20_1d = np.full_like(close_1d, np.nan)
    for i in range(19, len(close_1d)):
        std20_1d[i] = np.std(close_1d[i-19:i+1])
    
    # Align daily std to 12h timeframe
    std20_1d_aligned = align_htf_to_ltf(prices, df_1d, std20_1d)
    
    # Bollinger Bands: upper and lower (2 std dev from SMA)
    upper_bb_1d_aligned = sma10_1d_aligned + 2 * std20_1d_aligned
    lower_bb_1d_aligned = sma10_1d_aligned - 2 * std20_1d_aligned
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Conservative size to limit trades
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(sma10_1d_aligned[i]) or 
            np.isnan(std20_1d_aligned[i]) or
            np.isnan(upper_bb_1d_aligned[i]) or
            np.isnan(lower_bb_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current period volume vs 20-period average
        vol_ma_20 = np.full_like(volume, np.nan)
        for j in range(19, len(volume)):
            vol_ma_20[j] = np.mean(volume[j-19:j+1])
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: price touches lower Bollinger Band with volume surge
            if (close[i] <= lower_bb_1d_aligned[i] and 
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: price touches upper Bollinger Band with volume surge
            elif (close[i] >= upper_bb_1d_aligned[i] and 
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above SMA(10) or volume dries up
            if (close[i] > sma10_1d_aligned[i] or
                volume_ratio < 0.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below SMA(10) or volume dries up
            if (close[i] < sma10_1d_aligned[i] or
                volume_ratio < 0.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Bollinger_Touch_Volume"
timeframe = "12h"
leverage = 1.0