#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for ATR (HTF)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly ATR to daily timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Get daily data for price and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Donchian channels (20-period)
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Use previous day's Donchian levels (avoid look-ahead)
    donchian_high_1d_prev = np.roll(donchian_high_1d, 1)
    donchian_low_1d_prev = np.roll(donchian_low_1d, 1)
    donchian_high_1d_prev[0] = np.nan
    donchian_low_1d_prev[0] = np.nan
    
    # Align daily Donchian levels to daily timeframe (no shift needed as we already used prev)
    donchian_high_aligned = donchian_high_1d_prev
    donchian_low_aligned = donchian_low_1d_prev
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need Donchian, volume MA, and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(atr_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume_1d[i] > (1.5 * volume_ma20[i])
        # Volatility filter: ATR > 0 (always true, but keeps structure)
        volatility_filter = atr_1w_aligned[i] > 0
        
        if position == 0:
            # Long: price breaks above Donchian high with volume
            if (close_1d[i] > donchian_high_aligned[i] and volume_filter and volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume
            elif (close_1d[i] < donchian_low_aligned[i] and volume_filter and volatility_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below Donchian low or volatility drops
            if close_1d[i] < donchian_low_aligned[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above Donchian high or volatility drops
            if close_1d[i] > donchian_high_aligned[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian_Breakout_Volume_ATRFilter"
timeframe = "1d"
leverage = 1.0