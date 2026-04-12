#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_supertrend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Supertrend calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(10) on daily data
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend parameters
    atr_mult = 3.0
    
    # Basic upper and lower bands
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + (atr_mult * atr)
    lower_band = hl2 - (atr_mult * atr)
    
    # Initialize Supertrend
    supertrend = np.full_like(close_1d, np.nan, dtype=float)
    trend = np.ones_like(close_1d, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(10, len(close_1d)):
        if np.isnan(atr[i]) or atr[i] == 0:
            supertrend[i] = supertrend[i-1] if i > 0 else hl2[i]
            trend[i] = trend[i-1] if i > 0 else 1
            continue
            
        if close_1d[i] > upper_band[i-1]:
            trend[i] = 1
        elif close_1d[i] < lower_band[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            
        if trend[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1] if i > 0 else lower_band[i])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1] if i > 0 else upper_band[i])
    
    # Align Supertrend to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    trend_aligned = align_htf_to_ltf(prices, df_1d, trend)
    
    # Volume filter - 20-period average on 6h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(supertrend_aligned[i]) or np.isnan(trend_aligned[i]) or 
            np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend from Supertrend
        uptrend = trend_aligned[i] == 1
        downtrend = trend_aligned[i] == -1
        
        # Price relative to Supertrend line
        price_above_st = close[i] > supertrend_aligned[i]
        price_below_st = close[i] < supertrend_aligned[i]
        
        # Entry conditions with volume confirmation
        long_signal = uptrend and price_above_st and volume_ok[i]
        short_signal = downtrend and price_below_st and volume_ok[i]
        
        # Exit when trend changes
        exit_long = not uptrend  # Exit long when trend turns down
        exit_short = not downtrend  # Exit short when trend turns up
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals