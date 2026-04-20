#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Pivot_S1S2_Breakout_Volume_ATRFilter_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for pivot levels and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily average volume for spike detection (20-period) with vectorized approach
    vol_1d = df_1d['volume'].values
    vol_avg_1d = np.full(len(vol_1d), np.nan)
    for i in range(19, len(vol_1d)):
        vol_avg_1d[i] = np.mean(vol_1d[i-19:i+1])
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate ATR for stop loss (14-period on 4h data) with vectorized approach
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full_like(tr, np.nan)
    for i in range(13, len(tr)):
        atr[i] = np.nanmean(tr[i-13:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get previous completed daily bar for pivot calculation
        if len(df_1d) < 2:
            continue
            
        # Calculate daily pivot levels for each daily bar
        daily_high = df_1d['high'].values
        daily_low = df_1d['low'].values
        daily_close = df_1d['close'].values
        
        # Arrays to store daily S1 and S2 levels
        daily_s1 = np.full_like(daily_close, np.nan)
        daily_s2 = np.full_like(daily_close, np.nan)
        
        # Calculate for each daily bar (starting from index 1 to avoid look-ahead)
        for j in range(1, len(daily_close)):
            pivot = (daily_high[j-1] + daily_low[j-1] + daily_close[j-1]) / 3.0
            range_val = daily_high[j-1] - daily_low[j-1]
            if range_val > 0:
                daily_s1[j] = pivot - range_val
                daily_s2[j] = pivot - 2.0 * range_val
        
        # Align the daily S1/S2 to 4h timeframe
        daily_s1_aligned = align_htf_to_ltf(prices, df_1d, daily_s1)
        daily_s2_aligned = align_htf_to_ltf(prices, df_1d, daily_s2)
        
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        current_atr = atr[i]
        
        # Volume spike: current volume > 1.5x daily average volume
        vol_spike = (not np.isnan(vol_avg_1d_aligned[i]) and 
                     current_volume > 1.5 * vol_avg_1d_aligned[i])
        
        if position == 0:
            # Long: price breaks above daily S1 with volume spike
            if (not np.isnan(daily_s1_aligned[i]) and 
                current_close > daily_s1_aligned[i] and vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            # Short: price breaks below daily S2 with volume spike
            elif (not np.isnan(daily_s2_aligned[i]) and 
                  current_close < daily_s2_aligned[i] and vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: price breaks below daily S2 or ATR stop loss
            if (not np.isnan(daily_s2_aligned[i]) and 
                current_close < daily_s2_aligned[i]):
                signals[i] = 0.0
                position = 0
            elif current_atr > 0 and current_close < entry_price - 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above daily S1 or ATR stop loss
            if (not np.isnan(daily_s1_aligned[i]) and 
                current_close > daily_s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            elif current_atr > 0 and current_close > entry_price + 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals