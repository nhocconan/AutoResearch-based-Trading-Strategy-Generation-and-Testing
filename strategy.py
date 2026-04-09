#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h HMA(21) trend filter + volume confirmation
# - Long when price breaks above Donchian(20) high AND 12h HMA(21) is rising AND volume > 1.5x average
# - Short when price breaks below Donchian(20) low AND 12h HMA(21) is falling AND volume > 1.5x average
# - Uses ATR(14) for stoploss: exit when price moves 2.5x ATR against position
# - Fixed position size 0.25 to control drawdown
# - Works in both bull and bear markets by requiring volume confirmation and trend alignment
# - Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe

name = "4h_12h_donchian_hma_volume_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h HMA(21)
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    wma_half = np.array([wma(close_12h[i:i+half_len], half_len) if i+half_len <= len(close_12h) else np.nan 
                         for i in range(len(close_12h))])
    wma_full = np.array([wma(close_12h[i:i+21], 21) if i+21 <= len(close_12h) else np.nan 
                         for i in range(len(close_12h))])
    
    raw_hma = 2 * wma_half - wma_full
    hma_12h = np.array([wma(raw_hma[i:i+sqrt_len], sqrt_len) if i+sqrt_len <= len(raw_hma) else np.nan 
                        for i in range(len(raw_hma))])
    
    # Align HMA to 4h timeframe
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_12h_aligned[i]) or np.isnan(atr[i]) or atr[i] <= 0 or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Calculate HMA slope for trend direction
        if i >= 51:
            hma_slope = hma_12h_aligned[i] - hma_12h_aligned[i-1]
        else:
            hma_slope = 0
        
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            
            # ATR-based trailing stop: exit if price drops 2.5x ATR from highest high
            if close[i] < highest_high_since_entry - 2.5 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # ATR-based trailing stop: exit if price rises 2.5x ATR from lowest low
            if close[i] > lowest_low_since_entry + 2.5 * atr[i]:
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price breaks above Donchian high AND HMA rising AND volume confirmation
            if (close[i] > highest_high[i] and 
                hma_slope > 0 and 
                volume_confirm[i]):
                position = 1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low AND HMA falling AND volume confirmation
            elif (close[i] < lowest_low[i] and 
                  hma_slope < 0 and 
                  volume_confirm[i]):
                position = -1
                highest_high_since_entry = high[i]
                lowest_low_since_entry = low[i]
                signals[i] = -0.25
    
    return signals