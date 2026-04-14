#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d ADX filter + volume confirmation
# In trending markets (ADX > 25): breakout of 20-period high/low triggers entry
# In ranging markets (ADX < 20): no trades to avoid false breakouts
# Volume > 1.5x 20-period average confirms breakout strength
# Designed for 60-120 trades over 4 years with controlled frequency

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) - no look-ahead
    # Upper band: highest high of past 20 periods (excluding current)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    # Lower band: lowest low of past 20 periods (excluding current)
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (14-period) for trend filter
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
    
    for i in range(20, n):  # Start after Donchian calculation
        # Get aligned 1d ADX
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)[i]
        
        # Check for NaN values
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_1d_aligned)):
            continue
        
        # Breakout conditions
        bullish_breakout = close[i] > donchian_high[i]
        bearish_breakout = close[i] < donchian_low[i]
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Regime filter: only trade in trending markets
        trending = adx_1d_aligned > 25
        
        if position == 0:  # No position - look for entries
            if trending and volume_confirm:
                if bullish_breakout:
                    position = 1
                    signals[i] = position_size
                elif bearish_breakout:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit on reversal
            if close[i] < donchian_low[i]:  # Break below lower band
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit on reversal
            if close[i] > donchian_high[i]:  # Break above upper band
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian_1dADX_Volume"
timeframe = "6h"
leverage = 1.0