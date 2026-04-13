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
    
    # Get 4h data for primary timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data for Donchian channel calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channel (20-period)
    # Upper band = highest high over 20 days
    # Lower band = lowest low over 20 days
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian levels to 4h timeframe (completed 1d bars only)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    # ATR-based volatility filter: only trade when ATR(14) > 0.5 * ATR(50)
    # Ensures we trade in sufficient volatility regimes
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    vol_filter = (atr14 > 0.5 * atr50) & ~np.isnan(atr50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to ensure indicators are ready
        # Skip if data not ready
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(avg_volume[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > upper_band_aligned[i]  # break above upper band
        breakout_down = close[i] < lower_band_aligned[i]  # break below lower band
        
        # Entry conditions with volume and volatility filters
        long_entry = breakout_up and volume_confirmed[i] and vol_filter[i] and position != 1
        short_entry = breakout_down and volume_confirmed[i] and vol_filter[i] and position != -1
        
        # Exit conditions: reverse signal or midpoint retracement
        exit_long = (position == 1 and close[i] < (upper_band_aligned[i] + lower_band_aligned[i]) / 2)
        exit_short = (position == -1 and close[i] > (upper_band_aligned[i] + lower_band_aligned[i]) / 2)
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_volume_filter_v1"
timeframe = "4h"
leverage = 1.0