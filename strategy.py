#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_Supertrend_Breakout_Volume_v1"
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
    
    # Get daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR for Supertrend (daily)
    atr_period = 10
    tr1 = high_1d[1:] - low_1d[:-1]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Supertrend calculation (daily)
    factor = 3.0
    hl_avg = (high_1d + low_1d) / 2
    upper_band = hl_avg + factor * atr
    lower_band = hl_avg - factor * atr
    
    # Initialize Supertrend arrays
    supertrend = np.full_like(close_1d, np.nan)
    uptrend = np.full_like(close_1d, True)
    
    for i in range(1, len(close_1d)):
        if np.isnan(atr[i-1]) or np.isnan(close_1d[i-1]):
            continue
            
        if close_1d[i] > upper_band[i-1]:
            uptrend[i] = True
        elif close_1d[i] < lower_band[i-1]:
            uptrend[i] = False
        else:
            uptrend[i] = uptrend[i-1]
            if uptrend[i] and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if not uptrend[i] and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if uptrend[i]:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align Supertrend and trend direction to 6h
    supertrend_6h = align_htf_to_ltf(prices, df_1d, supertrend)
    uptrend_6h = align_htf_to_ltf(prices, df_1d, uptrend.astype(float))  # Convert bool to float for alignment
    
    # Calculate daily Donchian channels for breakout levels
    donchian_period = 20
    highest_high = pd.Series(high_1d).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low_1d).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Align Donchian levels to 6h
    highest_high_6h = align_htf_to_ltf(prices, df_1d, highest_high)
    lowest_low_6h = align_htf_to_ltf(prices, df_1d, lowest_low)
    
    # Volume confirmation: current volume > 1.8x 24-period average (4 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 24)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(supertrend_6h[i]) or np.isnan(uptrend_6h[i]) or \
           np.isnan(highest_high_6h[i]) or np.isnan(lowest_low_6h[i]) or \
           np.isnan(vol_ma_24[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        
        # Volume spike: current volume > 1.8x average
        volume_spike = vol > 1.8 * vol_ma
        
        # Trend filter: use daily Supertrend direction
        is_uptrend = uptrend_6h[i] > 0.5
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume spike and daily uptrend
            if price > highest_high_6h[i] and volume_spike and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume spike and daily downtrend
            elif price < lowest_low_6h[i] and volume_spike and not is_uptrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below Supertrend (trend reversal)
            if price < supertrend_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above Supertrend (trend reversal)
            if price > supertrend_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals