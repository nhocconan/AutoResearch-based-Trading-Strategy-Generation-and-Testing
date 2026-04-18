# [56806] Hypothesis: 4h Donchian breakout + volume spike + 1d ATR filter + time-of-day filter
# Targets breakouts with institutional volume in trending markets, avoids low-volatility periods.
# Works in bull/bear by only taking breaks in direction of higher timeframe trend (via ATR slope).
# Designed for low trade frequency: <50/year to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])  # align length
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR slope for trend filter (rising ATR = trending market)
    atr_slope = pd.Series(atr_14).diff(3)  # 3-period slope
    atr_slope_arr = atr_slope.values
    
    # Align ATR and slope to 4h
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_slope_aligned = align_htf_to_ltf(prices, df_1d, atr_slope_arr)
    
    # Donchian channel (20-period) on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Time-of-day filter: avoid low-liquidity hours (0-6 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    time_filter = (hours >= 6) & (hours <= 22)  # 6am-10pm UTC
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 100  # need enough history
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_slope_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_14_aligned[i]
        atr_slope_val = atr_slope_aligned[i]
        upper = high_20[i]
        lower = low_20[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume spike, rising ATR, and good time
            if (price > upper and 
                volume_spike[i] and 
                atr_slope_val > 0 and 
                time_filter[i]):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian with volume spike, rising ATR, and good time
            elif (price < lower and 
                  volume_spike[i] and 
                  atr_slope_val > 0 and 
                  time_filter[i]):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position: hold until reversal or ATR-based stop
            signals[i] = 0.25
            # Exit: price breaks below lower Donchian (reversal)
            if price < lower:
                signals[i] = 0.0
                position = 0
            # Optional: time-based exit (close position after 12 bars = 3 days)
            elif (i - entry_bar) >= 12:  # need to track entry bar
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: hold until reversal
            signals[i] = -0.25
            # Exit: price breaks above upper Donchian (reversal)
            if price > upper:
                signals[i] = 0.0
                position = 0
            # Optional: time-based exit
            elif (i - entry_bar) >= 12:
                signals[i] = 0.0
                position = 0
    
    return signals

# Fix: track entry bar for time-based exit
def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])  # align length
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR slope for trend filter (rising ATR = trending market)
    atr_slope = pd.Series(atr_14).diff(3)  # 3-period slope
    atr_slope_arr = atr_slope.values
    
    # Align ATR and slope to 4h
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_slope_aligned = align_htf_to_ltf(prices, df_1d, atr_slope_arr)
    
    # Donchian channel (20-period) on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Time-of-day filter: avoid low-liquidity hours (0-6 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    time_filter = (hours >= 6) & (hours <= 22)  # 6am-10pm UTC
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    entry_bar = -1  # track entry bar for time-based exit
    
    start_idx = 100  # need enough history
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_slope_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_14_aligned[i]
        atr_slope_val = atr_slope_aligned[i]
        upper = high_20[i]
        lower = low_20[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume spike, rising ATR, and good time
            if (price > upper and 
                volume_spike[i] and 
                atr_slope_val > 0 and 
                time_filter[i]):
                signals[i] = 0.25
                position = 1
                entry_price = price
                entry_bar = i
            # Short: price breaks below lower Donchian with volume spike, rising ATR, and good time
            elif (price < lower and 
                  volume_spike[i] and 
                  atr_slope_val > 0 and 
                  time_filter[i]):
                signals[i] = -0.25
                position = -1
                entry_price = price
                entry_bar = i
        
        elif position == 1:
            # Long position: hold until reversal or time-based exit
            signals[i] = 0.25
            # Exit: price breaks below lower Donchian (reversal)
            if price < lower:
                signals[i] = 0.0
                position = 0
            # Time-based exit: close after 12 bars (3 days)
            elif (i - entry_bar) >= 12:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: hold until reversal or time-based exit
            signals[i] = -0.25
            # Exit: price breaks above upper Donchian (reversal)
            if price > upper:
                signals[i] = 0.0
                position = 0
            # Time-based exit: close after 12 bars (3 days)
            elif (i - entry_bar) >= 12:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian_Breakout_Volume_Spike_ATRFilter_Time"
timeframe = "4h"
leverage = 1.0