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
    
    # === 1d Bollinger Bands (20,2) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate SMA20
    sma_20 = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 19:
            sma_20[i] = np.mean(close_1d[i-19:i+1])
        else:
            sma_20[i] = np.nan
    
    # Calculate standard deviation
    std_20 = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 19:
            std_20[i] = np.std(close_1d[i-19:i+1])
        else:
            std_20[i] = np.nan
    
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # === 1d Bollinger Band Width (for squeeze detection) ===
    bb_width = (upper_bb - lower_bb) / sma_20
    
    # === 1d BB Width percentile (20-period) for regime detection ===
    bb_width_percentile = np.full_like(bb_width, np.nan)
    for i in range(len(bb_width)):
        if i >= 19:
            window = bb_width[i-19:i+1]
            rank = np.sum(window <= bb_width[i]) / len(window)
            bb_width_percentile[i] = rank * 100
        else:
            bb_width_percentile[i] = np.nan
    
    # === Align indicators to 12h timeframe ===
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # === 1d Volume confirmation ===
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period average volume on 1d timeframe
    vol_ma_20 = np.full_like(volume_1d, np.nan)
    for i in range(len(volume_1d)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
        else:
            vol_ma_20[i] = np.nan
    
    # Volume confirmation: current 1d volume > 1.5x 20-period average
    vol_confirm = volume_1d > vol_ma_20 * 1.5
    
    # === 12h Session filter (08-20 UTC) ===
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if outside session
        if not session_filter[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Market regime: low volatility squeeze (BB Width < 20th percentile)
        is_squeeze = bb_width_percentile_aligned[i] < 20
        
        # Entry logic: only enter when flat AND volume confirmation
        if position == 0:
            # Long: BB squeeze + price near lower BB + volume confirmation
            if (is_squeeze and 
                close[i] <= lower_bb_aligned[i] * 1.02 and  # within 2% of lower BB
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: BB squeeze + price near upper BB + volume confirmation
            elif (is_squeeze and 
                  close[i] >= upper_bb_aligned[i] * 0.98 and  # within 2% of upper BB
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price reaches upper BB or squeeze breaks
            if (close[i] >= upper_bb_aligned[i] or 
                not is_squeeze):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches lower BB or squeeze breaks
            if (close[i] <= lower_bb_aligned[i] or 
                not is_squeeze):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_BB_Squeeze_Breakout_VolumeFilter"
timeframe = "12h"
leverage = 1.0