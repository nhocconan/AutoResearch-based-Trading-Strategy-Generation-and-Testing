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
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 50-period SMA of weekly close (trend filter)
    if len(close_1w) < 50:
        return np.zeros(n)
    
    sma50_1w = np.full_like(close_1w, np.nan)
    for i in range(49, len(close_1w)):
        sma50_1w[i] = np.mean(close_1w[i-49:i+1])
    
    # Align weekly SMA to daily timeframe
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # Calculate daily ATR(14) for volatility filter and stop
    tr = np.zeros_like(high)
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i],
                   abs(high[i] - high[i-1]),
                   abs(low[i] - low[i-1]))
    
    atr = np.full_like(high, np.nan)
    if len(high) >= 15:
        atr[14] = np.mean(tr[1:15])
        for i in range(15, len(high)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-period Donchian channels on daily
    upper_donchian = np.full_like(high, np.nan)
    lower_donchian = np.full_like(low, np.nan)
    
    for i in range(19, len(high)):
        upper_donchian[i] = np.max(high[i-19:i+1])
        lower_donchian[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period SMA of daily close (exit condition)
    sma20 = np.full_like(close, np.nan)
    for i in range(19, len(close)):
        sma20[i] = np.mean(close[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(sma50_1w_aligned[i]) or 
            np.isnan(atr[i]) or
            np.isnan(upper_donchian[i]) or
            np.isnan(lower_donchian[i]) or
            np.isnan(sma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current volume vs 20-period average
        vol_ma_20 = np.full_like(volume, np.nan)
        for j in range(19, len(volume)):
            vol_ma_20[j] = np.mean(volume[j-19:j+1])
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume surge and weekly uptrend
            if (close[i] > upper_donchian[i] and 
                volume_ratio > 2.0 and
                close[i] > sma50_1w_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian lower with volume surge and weekly downtrend
            elif (close[i] < lower_donchian[i] and 
                  volume_ratio > 2.0 and
                  close[i] < sma50_1w_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below 20-day SMA or volume dries up
            if (close[i] < sma20[i] or
                volume_ratio < 0.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above 20-day SMA or volume dries up
            if (close[i] > sma20[i] or
                volume_ratio < 0.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0