#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d ATR-based breakout and volume confirmation.
# Long: Price closes above (1d high + 0.5 * 14-day ATR) with volume > 1.3x 20-period average.
# Short: Price closes below (1d low - 0.5 * 14-day ATR) with volume > 1.3x 20-period average.
# Uses volatility-adjusted breakout levels from daily timeframe to filter noise.
# Volume confirmation ensures breakouts have institutional participation.
# Conservative position sizing (0.25) to manage drawdown in volatile markets.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for ATR and price levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-day ATR on daily timeframe
    tr1 = np.zeros(len(high_1d))
    tr2 = np.zeros(len(high_1d))
    tr3 = np.zeros(len(high_1d))
    tr1[1:] = np.abs(high_1d[1:] - low_1d[1:])
    tr2[1:] = np.abs(high_1d[1:] - close_1d[:-1])
    tr3[1:] = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = np.full(len(high_1d), np.nan)
    for i in range(14, len(tr)):
        atr_14[i] = np.mean(tr[i-13:i+1])
    
    # Calculate breakout levels: ±0.5 * ATR from daily high/low
    breakout_up = np.full(len(high_1d), np.nan)
    breakout_down = np.full(len(high_1d), np.nan)
    for i in range(len(high_1d)):
        if not np.isnan(atr_14[i]):
            breakout_up[i] = high_1d[i] + 0.5 * atr_14[i]
            breakout_down[i] = low_1d[i] - 0.5 * atr_14[i]
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d breakout levels to 12h
    breakout_up_aligned = align_htf_to_ltf(prices, df_1d, breakout_up)
    breakout_down_aligned = align_htf_to_ltf(prices, df_1d, breakout_down)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(breakout_up_aligned[i]) or np.isnan(breakout_down_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        up_level = breakout_up_aligned[i]
        down_level = breakout_down_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: price closes above breakout_up + volume confirmation
            if (price > up_level and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price closes below breakout_down + volume confirmation
            elif (price < down_level and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below breakout_down (opposite level)
            if price < down_level:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above breakout_up (opposite level)
            if price > up_level:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_ATR_Breakout_Volume"
timeframe = "12h"
leverage = 1.0