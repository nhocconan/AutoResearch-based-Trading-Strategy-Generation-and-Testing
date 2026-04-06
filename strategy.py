#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian Channel Breakout with Volume Confirmation and ATR Stop.
# Uses daily Donchian channels (20-period) calculated from prior day's OHLC.
# Breakout above upper channel or below lower channel triggers entries.
# Volume filter (current volume > 1.5x 20-period average) ensures quality signals.
# Works in bull/bear markets via breakout of established price channels.
# Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_donchian20_vol_atr_v2"
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
    
    # Daily OHLC for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period high/low)
    upper = np.full(len(high_1d), np.nan)
    lower = np.full(len(high_1d), np.nan)
    
    for i in range(19, len(high_1d)):
        upper[i] = np.max(high_1d[i-19:i+1])
        lower[i] = np.min(low_1d[i-19:i+1])
    
    # Align Donchian levels to 4h timeframe (shifted by 1 daily bar)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
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
            # Exit: price reaches lower Donchian (mean reversion) or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (close[i] <= lower_aligned[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches upper Donchian (mean reversion) or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (close[i] >= upper_aligned[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Breakout above upper Donchian (trend continuation)
                if (close[i] > upper_aligned[i] and close[i-1] <= upper_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Breakdown below lower Donchian (trend continuation)
                elif (close[i] < lower_aligned[i] and close[i-1] >= lower_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals