#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 1-week lookback high/low (for breakout)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly high/low from previous week (avoid look-ahead)
    high_1w_prev = np.roll(high_1w, 1)
    low_1w_prev = np.roll(low_1w, 1)
    high_1w_prev[0] = np.nan
    low_1w_prev[0] = np.nan
    
    # Align weekly levels to daily timeframe
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w_prev)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w_prev)
    
    # Volume confirmation: current volume > 1.8 * 20-day average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR filter to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 21  # Need 20-day MA and previous week data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma20[i]) or 
            np.isnan(high_1w_aligned[i]) or 
            np.isnan(low_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-day average
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        # Volatility filter: ATR > ATR MA20 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma20[i]
        
        if position == 0:
            # Long: price breaks above weekly high with volume and volatility
            if close[i] > high_1w_aligned[i] and volume_filter and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly low with volume and volatility
            elif close[i] < low_1w_aligned[i] and volume_filter and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below weekly low or volatility drops
            if close[i] < low_1w_aligned[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above weekly high or volatility drops
            if close[i] > high_1w_aligned[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyBreakout_Vol_VolatilityFilter"
timeframe = "1d"
leverage = 1.0