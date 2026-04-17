#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h volume confirmation and 1d ADX trend filter.
Long when price breaks above 20-bar high with volume > 1.3x 12h average volume AND 1d ADX > 25.
Short when price breaks below 20-bar low with volume > 1.3x 12h average volume AND 1d ADX > 25.
Exit when price touches the opposite Donchian level or ADX < 20 (trend weakening).
Uses 12h for volume confirmation and 1d for ADX trend filter.
Designed to capture strong trends in both bull and bear markets with volume confirmation.
Target: 12-25 trades/year per symbol (50-100 total over 4 years).
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
    
    # Get 12h data for volume MA
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Get 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(low_1d)
    tr = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(0, high_1d[i] - high_1d[i-1])
        minus_dm[i] = max(0, low_1d[i-1] - low_1d[i])
        tr[i] = max(high_1d[i] - low_1d[i], 
                    abs(high_1d[i] - close_1d[i-1]), 
                    abs(low_1d[i] - close_1d[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    tr14 = np.zeros_like(tr)
    plus_dm14 = np.zeros_like(plus_dm)
    minus_dm14 = np.zeros_like(minus_dm)
    
    # Initial values
    tr14[period] = np.nansum(tr[1:period+1])
    plus_dm14[period] = np.nansum(plus_dm[1:period+1])
    minus_dm14[period] = np.nansum(minus_dm[1:period+1])
    
    # Wilder smoothing
    for i in range(period+1, len(tr)):
        tr14[i] = tr14[i-1] - (tr14[i-1] / period) + tr[i]
        plus_dm14[i] = plus_dm14[i-1] - (plus_dm14[i-1] / period) + plus_dm[i]
        minus_dm14[i] = minus_dm14[i-1] - (minus_dm14[i-1] / period) + minus_dm[i]
    
    # Avoid division by zero
    dx = np.zeros_like(tr14)
    mask = tr14 != 0
    dx[mask] = (np.abs(plus_dm14[mask] - minus_dm14[mask]) / tr14[mask]) * 100
    
    # ADX is smoothed DX
    adx = np.zeros_like(dx)
    adx[2*period] = np.nanmean(dx[period:2*period]) if np.any(~np.isnan(dx[period:2*period])) else 0
    
    for i in range(2*period+1, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(50, 2*14+1)  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 12h average volume
        volume_confirmed = volume[i] > 1.3 * vol_ma_20_aligned[i]
        
        # ADX trend filter: ADX > 25 for strong trend
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20  # exit condition
        
        # Donchian channels (20-period)
        if i >= 20:
            donch_high = np.max(high[i-20:i])
            donch_low = np.min(low[i-20:i])
        else:
            donch_high = np.max(high[:i+1]) if i > 0 else high[i]
            donch_low = np.min(low[:i+1]) if i > 0 else low[i]
        
        # Breakout conditions
        breakout_high = close[i] > donch_high
        breakout_low = close[i] < donch_low
        
        if position == 0:
            # Long: break above Donchian high with volume confirmation and strong trend
            if (breakout_high and volume_confirmed and strong_trend):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume confirmation and strong trend
            elif (breakout_low and volume_confirmed and strong_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches Donchian low OR trend weakens
            if (close[i] <= donch_low) or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches Donchian high OR trend weakens
            if (close[i] >= donch_high) or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Volume_12h_1dADX_Trend"
timeframe = "6h"
leverage = 1.0