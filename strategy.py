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
    
    # Load 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h EMA on 1d close (trend filter)
    close_1d_series = pd.Series(close_1d)
    ema_12h_1d = close_1d_series.ewm(span=12, adjust=False, min_periods=12).values
    
    # Calculate 1d ATR for volatility filter
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(high_1d, 1)), 
                               np.abs(low_1d - np.roll(low_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).values
    
    # Align 1d indicators to 12h timeframe
    ema_12h_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_12h_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate Donchian channels on 1d (20-period)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Calculate volume ratio on 12h (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(ema_12h_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: breakout above 1d Donchian high + volume surge + above 12h EMA
            if (close[i] > donch_high_aligned[i] and 
                volume_ratio > 2.0 and
                close[i] > ema_12h_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: breakdown below 1d Donchian low + volume surge + below 12h EMA
            elif (close[i] < donch_low_aligned[i] and 
                  volume_ratio > 2.0 and
                  close[i] < ema_12h_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: close below 12h EMA or ATR-based trailing stop
            if (close[i] < ema_12h_1d_aligned[i] or
                close[i] < high[i] - 2.5 * atr_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: close above 12h EMA or ATR-based trailing stop
            if (close[i] > ema_12h_1d_aligned[i] or
                close[i] > low[i] + 2.5 * atr_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_EMA_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0