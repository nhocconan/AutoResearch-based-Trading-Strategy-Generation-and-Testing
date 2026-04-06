#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian Channel Breakout with Volume Confirmation and ATR Stoploss.
# Uses 20-period Donchian channels on 4h chart for trend direction.
# Entry on breakout above upper band (long) or below lower band (short) with volume > 1.5x 20-period average.
# Exit when price touches opposite band or stoploss hits (2x ATR).
# Works in bull/bear markets via breakout logic and strict risk management.
# Target: 75-200 trades over 4 years (19-50/year).

name = "4h_donchian20_vol_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(19, n):
        upper[i] = np.max(high[i-19:i+1])
        lower[i] = np.min(low[i-19:i+1])
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # ATR (14-period) for stoploss
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.full(n, np.nan)
    for i in range(13, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price touches lower band or stoploss
            stop_loss_level = entry_price - 2.0 * atr[i]
            
            if (close[i] <= lower[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price touches upper band or stoploss
            stop_loss_level = entry_price + 2.0 * atr[i]
            
            if (close[i] >= upper[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Breakout above upper band (long)
                if (close[i] > upper[i] and close[i-1] <= upper[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Breakdown below lower band (short)
                elif (close[i] < lower[i] and close[i-1] >= lower[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals