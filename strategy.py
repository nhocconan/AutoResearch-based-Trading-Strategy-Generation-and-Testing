#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1d volume and ATR filter
    # Enter long when price breaks above 20-period 12h high with volume > 1.5x 20-bar avg
    # Enter short when price breaks below 20-period 12h low with volume > 1.5x 20-bar avg
    # Exit when price crosses 12h ATR-based trailing stop
    # Uses 1d HTF for volume confirmation (more stable) and ATR calculation
    # Donchian breakouts capture strong momentum moves in both bull and bear markets
    # Volume confirmation reduces false breakouts
    # ATR trailing stop manages risk without being too tight
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for primary timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for volume and ATR confirmation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume average for confirmation
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to ensure indicators are ready
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_20[i]  # break above 20-period high
        breakout_down = close[i] < low_20[i]  # break below 20-period low
        
        # Volume and volatility confirmation
        volume_confirmed = volume[i] > (1.5 * avg_volume_1d_aligned[i])
        volatility_filter = atr_1d_aligned[i] > 0  # ensure ATR is valid
        
        # Entry conditions
        long_entry = breakout_up and volume_confirmed and volatility_filter and position != 1
        short_entry = breakout_down and volume_confirmed and volatility_filter and position != -1
        
        # ATR-based trailing stop exit
        exit_long = False
        exit_short = False
        if position == 1:
            # Trailing stop: highest high since entry minus 2*ATR
            # Simplified: exit if price drops below current close - 2*ATR
            exit_long = close[i] < (close_12h[i] - 2.0 * atr_1d_aligned[i])
        elif position == -1:
            # Trailing stop: lowest low since entry plus 2*ATR
            # Simplified: exit if price rises above current close + 2*ATR
            exit_short = close[i] > (close_12h[i] + 2.0 * atr_1d_aligned[i])
        
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

name = "12h_1d_donchian_volume_atr_v1"
timeframe = "12h"
leverage = 1.0