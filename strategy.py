#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d HMA21 trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves. 1d HMA21 provides smoother trend than EMA.
# Volume confirmation (>1.5x 20 EMA) reduces false breakouts. Discrete sizing 0.25 balances risk/reward.
# Works in bull/bear: trend filter prevents counter-trend entries. Target: 75-200 trades over 4 years.

name = "4h_Donchian20_1dHMA21_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1d HMA21 for trend direction
    close_1d = pd.Series(df_1d['close'])
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_n = 21 // 2
    sqrt_n = int(np.sqrt(21))
    wma_half = close_1d.ewm(span=half_n, adjust=False).mean()
    wma_full = close_1d.ewm(span=21, adjust=False).mean()
    raw_hma = 2 * wma_half - wma_full
    hma_21_1d = raw_hma.ewm(span=sqrt_n, adjust=False).mean().values
    
    # Align 1d HMA21 to 4h timeframe (completed 1d bar only)
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate Donchian(20) channels on 4h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(hma_21_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper + uptrend + volume spike
            if close[i] > highest_high[i] and close[i] > hma_21_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower + downtrend + volume spike
            elif close[i] < lowest_low[i] and close[i] < hma_21_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR trend changes OR volume drops
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if (close[i] < midpoint or 
                close[i] < hma_21_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR trend changes OR volume drops
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if (close[i] > midpoint or 
                close[i] > hma_21_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals