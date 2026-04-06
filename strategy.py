#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian Channel Breakout with Daily EMA Trend Filter and Volume Confirmation
# Uses 20-period Donchian channels on 12h timeframe for breakout signals.
# Filters: Price must be above/below daily EMA(40) for trend direction.
# Volume filter: Current volume > 1.3x 20-period average to ensure conviction.
# Works in bull/bear markets by trading breakouts in direction of higher timeframe trend.
# Target: 75-150 total trades over 4 years (19-38/year).

name = "12h_donchian20_1d_ema40_vol_v2"
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
    
    # Daily EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(40) on daily close
    ema_40 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 40:
        ema_40[39] = np.mean(close_1d[:40])
        for i in range(40, len(close_1d)):
            ema_40[i] = (close_1d[i] * 0.04878) + (ema_40[i-1] * 0.95122)
    
    # Align daily EMA to 12h timeframe
    ema_40_aligned = align_htf_to_ltf(prices, df_1d, ema_40)
    
    # Donchian channels (20-period) on 12h
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(19, n):
        upper[i] = np.max(high[i-19:i+1])
        lower[i] = np.min(low[i-19:i+1])
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_40_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches lower Donchian or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (close[i] <= lower[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches upper Donchian or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (close[i] >= upper[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long breakout: price breaks above upper Donchian AND above daily EMA
                if (close[i] > upper[i] and close[i-1] <= upper[i] and 
                    close[i] > ema_40_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown: price breaks below lower Donchian AND below daily EMA
                elif (close[i] < lower[i] and close[i-1] >= lower[i] and 
                      close[i] < ema_40_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals