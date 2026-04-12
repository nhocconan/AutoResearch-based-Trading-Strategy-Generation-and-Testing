#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_donchian_breakout_volume_v1
# Donchian(20) breakout on 4h timeframe confirmed by 1d volume surge and price > 1d VWAP.
# Works in bull markets via breakouts and in bear markets via short breakdowns.
# Volume confirmation reduces false breakouts. Target: 20-40 trades/year.
name = "4h_1d_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume surge and VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d VWAP (volume-weighted average price)
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    vwap_1d = (typical_price_1d * df_1d['volume'].values).cumsum() / df_1d['volume'].values.cumsum()
    vwap_1d = vwap_1d.values
    
    # Calculate 1d volume average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if 1d indicators not ready
        if np.isnan(vwap_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume surge condition: current 4h volume > 1.5x 1d average volume
        volume_surge = volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        # Breakout conditions
        bullish_breakout = (close[i] > highest_high[i-1]) and volume_surge and (close[i] > vwap_1d_aligned[i])
        bearish_breakout = (close[i] < lowest_low[i-1]) and volume_surge and (close[i] < vwap_1d_aligned[i])
        
        # Exit conditions: opposite breakout or loss of momentum
        exit_long = (close[i] < lowest_low[i-1]) and volume_surge
        exit_short = (close[i] > highest_high[i-1]) and volume_surge
        
        if bullish_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals