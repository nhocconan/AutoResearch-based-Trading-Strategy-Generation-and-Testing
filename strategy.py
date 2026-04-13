#!/usr/bin/env python3
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
    
    # Daily data for 4-hour analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 10-day Donchian channel on daily for tighter entries
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    donchian_low = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    
    # Calculate daily ATR for volatility filter
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]]))
    low_close = np.abs(low_1d - np.concatenate([[low_1d[0]], low_1d[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate daily volume and its 10-period average
    volume_1d = df_1d['volume'].values
    volume_ma_10_1d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    
    # Align all data to 4-hour timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    atr_10_aligned = align_htf_to_ltf(prices, df_1d, atr_10)
    volume_ma_10_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_10_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_10_aligned[i]) or np.isnan(volume_ma_10_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 4h volume > 1.5x daily volume MA (adjusted for 4h)
        # 6 4h periods per day, so daily MA/6 = approximate 4h period MA
        volume_4h_approx_ma = volume_ma_10_1d_aligned[i] / 6
        volume_condition = volume[i] > (volume_4h_approx_ma * 1.5)
        
        # Volatility filter: require sufficient volatility for breakout
        vol_condition = atr_10_aligned[i] > 0  # Always true if ATR calculated
        
        # Entry conditions: Donchian breakout with volume and volatility confirmation
        breakout_long = close[i] > donchian_high_aligned[i]
        breakout_short = close[i] < donchian_low_aligned[i]
        
        if position == 0:
            if breakout_long and volume_condition:
                position = 1
                signals[i] = position_size
            elif breakout_short and volume_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price breaks below Donchian low
            if close[i] < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price breaks above Donchian high
            if close[i] > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian_Breakout_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0