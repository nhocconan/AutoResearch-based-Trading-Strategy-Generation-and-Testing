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
    
    # Get daily data for ATR and moving averages
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14)
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 40-day EMA for trend filter
    ema_40_1d = pd.Series(close_1d).ewm(span=40, adjust=False, min_periods=40).mean().values
    
    # Calculate 20-day EMA for dynamic channel
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align daily indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_40_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_40_1d)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate upper and lower channels (EMA20 ± ATR)
    upper_channel = ema_20_1d_aligned + (1.5 * atr_1d_aligned)
    lower_channel = ema_20_1d_aligned - (1.5 * atr_1d_aligned)
    
    # Volume filter: current volume > 1.8 * 20-period average (4h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for daily indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_40_1d_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        if position == 0:
            # Long entry: price crosses above upper channel in uptrend with volume
            if close[i] > upper_channel[i] and close[i] > ema_40_1d_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price crosses below lower channel in downtrend with volume
            elif close[i] < lower_channel[i] and close[i] < ema_40_1d_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA20 or trend changes
            if close[i] < ema_20_1d_aligned[i] or close[i] < ema_40_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA20 or trend changes
            if close[i] > ema_20_1d_aligned[i] or close[i] > ema_40_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_EMAChannel_ATR_VolumeFilter"
timeframe = "4h"
leverage = 1.0