#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Channels with Volume and ATR Volatility Filter.
# Uses daily Donchian channels (20-day high/low) for trend identification.
# Enters long when price breaks above daily upper band with volume > 1.5x 20-period average.
# Enters short when price breaks below daily lower band with volume filter.
# Uses ATR-based stoploss (2x ATR) and exits at opposite band.
# Works in bull/bear markets via breakout logic with volatility filtering.
# Target: 50-150 trades over 4 years (12-37/year).

name = "12h_donchian_vol_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily OHLC for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period high/low)
    upper = np.full(len(high_1d), np.nan)
    lower = np.full(len(low_1d), np.nan)
    
    for i in range(19, len(high_1d)):
        upper[i] = np.max(high_1d[i-19:i+1])
        lower[i] = np.min(low_1d[i-19:i+1])
    
    # Align Donchian levels to 12h timeframe (shifted by 1 daily bar)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    # ATR calculation for volatility and stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = 0.9 * atr[i-1] + 0.1 * tr[i]  # Wilder's smoothing
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if Donchian data not available
        if np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches lower band (opposite) or stoploss
            stop_loss_level = entry_price - 2.0 * atr[i]
            
            if (close[i] <= lower_aligned[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches upper band (opposite) or stoploss
            stop_loss_level = entry_price + 2.0 * atr[i]
            
            if (close[i] >= upper_aligned[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Breakout above upper band
                if close[i] > upper_aligned[i] and close[i-1] <= upper_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Breakdown below lower band
                elif close[i] < lower_aligned[i] and close[i-1] >= lower_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals