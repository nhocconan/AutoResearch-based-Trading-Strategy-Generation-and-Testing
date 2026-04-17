#!/usr/bin/env python3
"""
12h Turtle-style Breakout with Volume Confirmation and 1D EMA Trend Filter
Long: Price breaks above 1D Donchian high (20-day) + volume > 1.5x 12h volume MA + price > 1D EMA50
Short: Price breaks below 1D Donchian low (20-day) + volume > 1.5x 12h volume MA + price < 1D EMA50
Exit: Opposite break of 1D Donchian level
Focus: Breakout trading with trend filter to avoid false signals in chop
Target: 20-30 trades/year per symbol
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
    
    # Get 1D data for Donchian channels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 20-period Donchian channels on daily
    donch_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # 50-period EMA on daily for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h volume moving average (24-period for confirmation)
    df_12h = get_htf_data(prices, '12h')
    volume_ma_24 = pd.Series(df_12h['volume']).rolling(window=24, min_periods=24).mean().values
    volume_ma_24_12h = align_htf_to_ltf(prices, df_12h, volume_ma_24)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 60  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_24_12h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_24_12h[i]
        
        if position == 0:
            # Long: break above 1D Donchian high + volume + 1D trend
            if price > donch_high_aligned[i] and vol > 1.5 * vol_ma and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below 1D Donchian low + volume + 1D trend
            elif price < donch_low_aligned[i] and vol > 1.5 * vol_ma and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: break below 1D Donchian low
            if price < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above 1D Donchian high
            if price > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Volume_1DEMA50"
timeframe = "12h"
leverage = 1.0