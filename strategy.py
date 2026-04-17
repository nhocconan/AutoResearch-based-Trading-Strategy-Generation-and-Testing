#!/usr/bin/env python3
"""
4h_4020DonchianBreakout_1dTrendVolume_SL
Long: Price breaks above 40-bar Donchian high + 1d EMA50 uptrend + volume > 1.5x 4h volume SMA(20)
Short: Price breaks below 20-bar Donchian low + 1d EMA50 downtrend + volume > 1.5x 4h volume SMA(20)
Exit: Opposite Donchian breakout or price closes below/above 2-bar Donchian midpoint
Stop: ATR-based trailing stop (not implemented in signal - handled by exit conditions)
Designed to capture medium-term trends with institutional-grade breakout confirmation.
Target: 80-180 total trades over 4 years (20-45/year)
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 40-bar Donchian high (for long entry)
    donch_high_40 = pd.Series(high).rolling(window=40, min_periods=40).max().values
    
    # Calculate 20-bar Donchian low (for short entry)
    donch_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 2-bar Donchian midpoint (for exit)
    donch_high_2 = pd.Series(high).rolling(window=2, min_periods=2).max().values
    donch_low_2 = pd.Series(low).rolling(window=2, min_periods=2).min().values
    donch_mid_2 = (donch_high_2 + donch_low_2) / 2
    
    # Calculate 4h volume SMA(20)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(40, 20)  # need Donchian channels
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_40[i]) or np.isnan(donch_low_20[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_20[i]
        ema50_val = ema50_1d_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 40-bar Donchian high + 1d EMA50 uptrend + volume spike
            if (price > donch_high_40[i] and 
                ema50_val > ema50_1d_aligned[i-1] and 
                vol > 1.5 * vol_sma_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-bar Donchian low + 1d EMA50 downtrend + volume spike
            elif (price < donch_low_20[i] and 
                  ema50_val < ema50_1d_aligned[i-1] and 
                  vol > 1.5 * vol_sma_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price closes below 2-bar Donchian midpoint or opposite breakout
            if price < donch_mid_2[i] or price < donch_low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price closes above 2-bar Donchian midpoint or opposite breakout
            if price > donch_mid_2[i] or price > donch_high_40[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_4020DonchianBreakout_1dTrendVolume_SL"
timeframe = "4h"
leverage = 1.0