#!/usr/bin/env python3
"""
Hypothesis: 4h-based strategy using 12h Supertrend (ATR=10, multiplier=3) as primary trend filter, 
combined with 1d Donchian(20) breakout for entry and volume confirmation. 
Supertrend avoids whipsaws in sideways markets, Donchian breakouts capture momentum, 
and volume ensures conviction. Designed for 20-30 trades/year to minimize fee drift.
Works in bull markets (buy upper band breakout in uptrend) and bear markets 
(sell lower band breakout in downtrend).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Supertrend calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR(10) on 12h
    atr_12h = np.full(len(close_12h), np.nan)
    tr_12h = np.maximum(
        high_12h[1:] - low_12h[1:],
        np.maximum(
            np.abs(high_12h[1:] - close_12h[:-1]),
            np.abs(low_12h[1:] - close_12h[:-1])
        )
    )
    if len(tr_12h) >= 10:
        atr_12h[9] = np.mean(tr_12h[:10])
        for i in range(10, len(tr_12h)):
            atr_12h[i] = (atr_12h[i-1] * 9 + tr_12h[i]) / 10
    
    # Calculate Supertrend on 12h
    hl2_12h = (high_12h + low_12h) / 2
    upper_band_12h = hl2_12h + 3 * atr_12h
    lower_band_12h = hl2_12h - 3 * atr_12h
    
    supertrend_12h = np.full(len(close_12h), np.nan)
    direction_12h = np.full(len(close_12h), np.nan)  # 1 for uptrend, -1 for downtrend
    
    if len(close_12h) >= 11:
        # Initialize
        supertrend_12h[10] = upper_band_12h[10]
        direction_12h[10] = 1
        
        for i in range(11, len(close_12h)):
            # Update bands
            if close_12h[i-1] > supertrend_12h[i-1]:
                upper_band_12h[i] = min(upper_band_12h[i], upper_band_12h[i-1])
            else:
                upper_band_12h[i] = upper_band_12h[i]
                
            if close_12h[i-1] < supertrend_12h[i-1]:
                lower_band_12h[i] = max(lower_band_12h[i], lower_band_12h[i-1])
            else:
                lower_band_12h[i] = lower_band_12h[i]
            
            # Determine trend
            if close_12h[i] > supertrend_12h[i-1]:
                direction_12h[i] = 1
                supertrend_12h[i] = lower_band_12h[i]
            elif close_12h[i] < supertrend_12h[i-1]:
                direction_12h[i] = -1
                supertrend_12h[i] = upper_band_12h[i]
            else:
                direction_12h[i] = direction_12h[i-1]
                supertrend_12h[i] = supertrend_12h[i-1]
                if direction_12h[i] == 1 and lower_band_12h[i] < supertrend_12h[i]:
                    supertrend_12h[i] = lower_band_12h[i]
                if direction_12h[i] == -1 and upper_band_12h[i] > supertrend_12h[i]:
                    supertrend_12h[i] = upper_band_12h[i]
    
    # Align Supertrend direction to 4h timeframe
    direction_12h_4h = align_htf_to_ltf(prices, df_12h, direction_12h)
    
    # Get 1d data for Donchian(20) calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels (20-period) on 1d
    donchian_high_1d = np.full(len(high_1d), np.nan)
    donchian_low_1d = np.full(len(low_1d), np.nan)
    
    for i in range(20, len(high_1d)):
        donchian_high_1d[i] = np.max(high_1d[i-20:i])
        donchian_low_1d[i] = np.min(low_1d[i-20:i])
    
    # Align Donchian levels to 4h timeframe
    donchian_high_1d_4h = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_4h = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 20)  # need Donchian, Supertrend, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(direction_12h_4h[i]) or np.isnan(donchian_high_1d_4h[i]) or 
            np.isnan(donchian_low_1d_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: close above Donchian high with volume and uptrend
            if (close[i] > donchian_high_1d_4h[i] and 
                vol_confirmed and 
                direction_12h_4h[i] == 1):
                signals[i] = 0.25
                position = 1
            # Short entry: close below Donchian low with volume and downtrend
            elif (close[i] < donchian_low_1d_4h[i] and 
                  vol_confirmed and 
                  direction_12h_4h[i] == -1):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: close below Donchian low or reverse signal
            if close[i] < donchian_low_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above Donchian high or reverse signal
            if close[i] > donchian_high_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Supertrend12h_Donchian1d_Volume"
timeframe = "4h"
leverage = 1.0