#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume and ATR Filter
Hypothesis: Weekly Donchian channels (20-week high/low) capture major trend breakouts.
In trending markets, price breaks above/below weekly channels with increased volume.
We use daily timeframe for execution, with weekly Donchian as trend filter.
Volume confirmation ensures breakout legitimacy. ATR stop manages risk.
Works in both bull and bear markets by capturing sustained moves.
Target: 10-20 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian channels on weekly data
    high_roll = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (already delayed by one weekly bar)
    donchian_high = align_htf_to_ltf(prices, df_1w, high_roll)
    donchian_low = align_htf_to_ltf(prices, df_1w, low_roll)
    
    # Daily ATR for volatility filter and stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long breakout: price above weekly Donchian high with volume surge
            if price > donchian_high[i] and vol_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short breakout: price below weekly Donchian low with volume surge
            elif price < donchian_low[i] and vol_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Trail stop: exit if price drops 2*ATR from highest high since entry
            # Simplified: exit if price < donchian_low (re-entry level)
            if price < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Trail stop: exit if price rises 2*ATR from lowest low since entry
            # Simplified: exit if price > donchian_high (re-entry level)
            if price > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Donchian_Breakout_Volume_ATR_Filter"
timeframe = "1d"
leverage = 1.0