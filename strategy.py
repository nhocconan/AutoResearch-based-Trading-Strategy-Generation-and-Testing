#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyDonchian20_BullishPhase_ExitOnClose"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for Donchian channel (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Daily data for bullish phase confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Weekly Donchian(20): high/low of last 20 weekly candles
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly rolling high/low with period=20
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Bullish phase: price above weekly Donchian middle (mean of 20-period high/low)
    donchian_mid = (high_20 + low_20) / 2.0
    bullish_phase = high_1w > donchian_mid  # Weekly close above midpoint = bullish
    
    # Align weekly bullish phase to 6h timeframe
    bullish_phase_aligned = align_htf_to_ltf(prices, df_1w, bullish_phase)
    
    # Daily close price for trend alignment (optional filter)
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    start_idx = 100  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(bullish_phase_aligned[i]) or 
            np.isnan(sma_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: weekly bullish phase AND price above daily SMA50
            if bullish_phase_aligned[i] and price > sma_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Exit long: weekly bullish phase ends (weekly close below Donchian midpoint)
            # OR price closes below daily SMA50 (defensive exit)
            if not bullish_phase_aligned[i] or price < sma_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals