#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_Volume_Confirm_TrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily trend filter: EMA34 on daily close
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Donchian channels (20-period) on 4h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_channel = high_roll[i]
        lower_channel = low_roll[i]
        trend_filter = ema34_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: break above upper channel with volume and above daily EMA34
            if close_val > upper_channel and vol_filter and close_val > trend_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel with volume and below daily EMA34
            elif close_val < lower_channel and vol_filter and close_val < trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below lower channel
            if close_val < lower_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above upper channel
            if close_val > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals