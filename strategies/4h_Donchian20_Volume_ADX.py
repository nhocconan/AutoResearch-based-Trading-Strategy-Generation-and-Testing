#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + ADX trend filter
# Long when price breaks above Donchian upper band + volume spike + ADX > 25
# Short when price breaks below Donchian lower band + volume spike + ADX > 25
# Exit when price crosses opposite Donchian band or ADX < 20
# Designed for moderate trade frequency (~20-40/year) with edge in trending markets
# Works in both bull (strong uptrends) and bear (strong downtrends) markets

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 20-period Donchian channels
    upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # ADX calculation (14-period)
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h),
                       np.maximum(high_4h - np.roll(high_4h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)),
                        np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    atr_4h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_smooth / atr_4h
    minus_di = 100 * dm_minus_smooth / atr_4h
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0,
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx_4h = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align to 15m
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        adx_val = adx_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper band + volume spike + ADX > 25
            if price > upper and vol_spike and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band + volume spike + ADX > 25
            elif price < lower and vol_spike and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses opposite band or ADX < 20
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below lower band or trend weakens
                if price < lower or adx_val < 20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above upper band or trend weakens
                if price > upper or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Volume_ADX"
timeframe = "4h"
leverage = 1.0