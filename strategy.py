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
    
    # Get weekly data for weekly high/low (1w)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Align weekly high/low to daily timeframe (use previous week's levels)
    weekly_high = align_htf_to_ltf(prices, df_1w, high_1w)
    weekly_low = align_htf_to_ltf(prices, df_1w, low_1w)
    
    # Daily ATR for volatility filtering
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: current volume > 2.0 * 20-day average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need sufficient data for volume MA and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        if position == 0:
            # Long: price breaks above weekly high with volume confirmation
            if close[i] > weekly_high[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly low with volume confirmation
            elif close[i] < weekly_low[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below weekly low OR volatility drops significantly
            if close[i] < weekly_low[i] or (atr[i] < 0.5 * np.nanmedian(atr[max(0, i-20):i+1]) and i >= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above weekly high OR volatility drops significantly
            if close[i] > weekly_high[i] or (atr[i] < 0.5 * np.nanmedian(atr[max(0, i-20):i+1]) and i >= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyHighLow_Breakout_VolumeFilter"
timeframe = "1d"
leverage = 1.0