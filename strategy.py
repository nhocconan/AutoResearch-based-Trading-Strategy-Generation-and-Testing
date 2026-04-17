#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 12h Donchian(20) breakout + volume confirmation + trend filter (price > 4h EMA50).
Long when price breaks above 12h Donchian upper channel with volume confirmation and price > 4h EMA50 (uptrend).
Short when price breaks below 12h Donchian lower channel with volume confirmation and price < 4h EMA50 (downtrend).
Exit when price returns to the 12h Donchian midpoint (mean reversion to channel center).
Uses EMA50 for tighter trend filter vs EMA34 to reduce trade frequency and improve Sharpe.
Designed to capture medium-term breakouts with institutional volume while avoiding false breakouts in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channel calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Donchian(20) channels
    lookback = 20
    upper_12h = pd.Series(high_12h).rolling(window=lookback, min_periods=lookback).max().values
    lower_12h = pd.Series(low_12h).rolling(window=lookback, min_periods=lookback).min().values
    mid_12h = (upper_12h + lower_12h) / 2.0
    
    # Calculate 4h EMA50 for trend filter (tighter than EMA34)
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h Donchian levels to 4h timeframe
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    mid_12h_aligned = align_htf_to_ltf(prices, df_12h, mid_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_12h_aligned[i]) or 
            np.isnan(lower_12h_aligned[i]) or 
            np.isnan(mid_12h_aligned[i]) or 
            np.isnan(ema50[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 12h Donchian upper with volume and uptrend (price > EMA50)
            if (close[i] > upper_12h_aligned[i] and 
                volume_confirmed and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian lower with volume and downtrend (price < EMA50)
            elif (close[i] < lower_12h_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below 12h Donchian midpoint
            if close[i] <= mid_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above 12h Donchian midpoint
            if close[i] >= mid_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12hDonchian20_Breakout_Volume_EMA50_Trend"
timeframe = "4h"
leverage = 1.0