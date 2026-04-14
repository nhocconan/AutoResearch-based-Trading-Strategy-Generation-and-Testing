#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout + 1d ADX regime + volume confirmation
# Long when price breaks above Donchian(20) high in trending (ADX>25) or mean-reverts from lower band in ranging (ADX<20)
# Short when price breaks below Donchian(20) low in trending or mean-reverts from upper band in ranging
# Volume > 1.5x 20-period average for confirmation
# Target: 50-150 trades over 4 years with controlled frequency

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (14-period) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for ADX
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Get aligned 1d ADX
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)[i]
        
        # Check for NaN values
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_1d_aligned)):
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:  # No position - look for entries
            if volume_confirm:
                # Trending market (ADX > 25): breakout follow
                if adx_1d_aligned > 25:
                    if close[i] > highest_high[i]:
                        position = 1
                        signals[i] = position_size
                    elif close[i] < lowest_low[i]:
                        position = -1
                        signals[i] = -position_size
                # Ranging market (ADX < 20): mean reversion from bands
                elif adx_1d_aligned < 20:
                    if close[i] < lowest_low[i]:  # Price at lower band, buy
                        position = 1
                        signals[i] = position_size
                    elif close[i] > highest_high[i]:  # Price at upper band, sell
                        position = -1
                        signals[i] = -position_size
        elif position == 1:  # Long position - exit when price returns to midpoint or opposite band
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < midpoint:  # Return to midpoint
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when price returns to midpoint
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > midpoint:  # Return to midpoint
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian_1dADX_Regime_Volume"
timeframe = "12h"
leverage = 1.0