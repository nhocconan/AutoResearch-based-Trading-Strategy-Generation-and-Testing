#!/usr/bin/env python3
"""
4h_1d_adaptive_breakout_v1
Hypothesis: Adaptive breakout strategy using 1-day ATR-based channels to capture trends in both bull and bear markets.
Uses 1-day ATR to dynamically set breakout levels, with volume confirmation and volatility filter.
Designed for moderate trade frequency (~20-35 trades/year) to minimize fee decay.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_adaptive_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1-day ATR-based channels (avoid look-ahead)
    atr_period = 10
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1-day
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    atr_1d = pd.Series(tr_1d).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Dynamic channels: ATR multiplier based on volatility regime
    atr_mult = np.where(atr_1d > np.percentile(atr_1d, 70), 1.5, 1.0)  # Higher mult in high vol
    
    # Upper and lower bands (previous day's close ± ATR*mult)
    upper_band = np.roll(close_1d, 1) + atr_1d * atr_mult
    lower_band = np.roll(close_1d, 1) - atr_1d * atr_mult
    
    # Align bands to 4h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Volume confirmation: 1-day volume > 20-period average
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # 4h ATR for volatility filter
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_avg_20 = pd.Series(atr_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 20 to ensure sufficient data
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(atr_4h[i]) or 
            np.isnan(atr_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 1-day volume (aligned)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_confirm = vol_1d_current > vol_avg_20_aligned[i]
        
        # Volatility filter: only trade when ATR > 20-period average
        vol_filter = atr_4h[i] > atr_avg_20[i]
        
        price = close[i]
        upper = upper_band_aligned[i]
        lower = lower_band_aligned[i]
        
        # Entry conditions: Breakout of adaptive bands with volume and volatility
        long_signal = vol_confirm and vol_filter and (price > upper)
        short_signal = vol_confirm and vol_filter and (price < lower)
        
        # Exit conditions: Return to middle (previous day's close) or opposite band
        prev_close_aligned = align_htf_to_ltf(prices, df_1d, np.roll(close_1d, 1))[i]
        long_exit = price < prev_close_aligned
        short_exit = price > prev_close_aligned
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals