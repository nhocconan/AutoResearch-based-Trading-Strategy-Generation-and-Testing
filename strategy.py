#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout with 1d ATR volatility filter and volume confirmation.
    # Donchian(20) provides robust trend-following structure in both bull and bear markets.
    # 1d ATR filter ensures we only trade during volatility expansion (avoids chop).
    # Volume confirmation ensures institutional participation.
    # Target: 75-200 total trades over 4 years = 19-50/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and volume MA (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR(14) for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d)
    tr3 = np.abs(low_1d - close_1d)
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR mean (20-period) for regime filter
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d volume MA(20) for confirmation
    volume_1d = df_1d['volume'].values
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(atr_ma_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR > 20-period mean (expanding volatility)
        volatility_filter = atr_aligned[i] > atr_ma_aligned[i]
        
        # Volume filter: current volume > 20-period MA
        volume_filter = volume[i] > volume_ma_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = close[i] > high_roll[i]  # Break above upper band
        breakout_short = close[i] < low_roll[i]  # Break below lower band
        
        # Entry conditions: breakout with volatility AND volume filters
        long_entry = breakout_long and volatility_filter and volume_filter
        short_entry = breakout_short and volatility_filter and volume_filter
        
        # Exit conditions: price returns to opposite Donchian level
        long_exit = close[i] < low_roll[i]
        short_exit = close[i] > high_roll[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
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

name = "4h_1d_donchian_atr_volume_v1"
timeframe = "4h"
leverage = 1.0