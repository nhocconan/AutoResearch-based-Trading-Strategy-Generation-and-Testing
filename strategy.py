#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + volume confirmation + 12h EMA34 trend filter + ATR-based stoploss.
Long when price breaks above Donchian(20) high with volume > 1.5x 20-bar average and price > 12h EMA34.
Short when price breaks below Donchian(20) low with volume > 1.5x 20-bar average and price < 12h EMA34.
Exit on opposite Donchian breakout or when price closes below/above 12h EMA34.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-200 total trades over 4 years.
"""

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
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h EMA34
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for Donchian(20) and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(ema34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = 1.5 * vol_ma[i]
        ema34_val = ema34_12h_aligned[i]
        
        # Breakout conditions
        breakout_up = price > highest_high[i-1]  # Use previous bar's Donchian high
        breakout_down = price < lowest_low[i-1]   # Use previous bar's Donchian low
        
        if position == 0:
            # Long: bullish breakout with volume confirmation and uptrend filter
            if breakout_up and vol > vol_threshold and price > ema34_val:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout with volume confirmation and downtrend filter
            elif breakout_down and vol > vol_threshold and price < ema34_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish breakout or price closes below 12h EMA34
            if breakout_down or price < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish breakout or price closes above 12h EMA34
            if breakout_up or price > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_12hEMA34_Trend"
timeframe = "4h"
leverage = 1.0