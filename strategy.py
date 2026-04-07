#!/usr/bin/env python3
"""
6h_williams_vix_fix_12h_volume_v1
Hypothesis: Williams VIX Fix identifies volatility spikes and mean reversion opportunities. Combined with 12h volume confirmation and 12h EMA trend filter, this strategy captures reversals during high volatility periods while avoiding chop. Works in both bull and bear markets by fading extreme volatility spikes. Target: 50-150 total trades over 4 years (12-37/year) with strict entry conditions to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_williams_vix_fix_12h_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams VIX Fix on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 22:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams VIX Fix: wvf = ((highest(high, n) - low) / highest(high, n)) * 100
    # Using 22-period lookback as in original indicator
    highest_high = np.maximum.accumulate(high_12h)
    wvf = ((highest_high - low_12h) / highest_high) * 100
    
    # Calculate EMA trend filter on 12h
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate volume moving average on 12h
    vol_ma_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all 12h indicators to 6h timeframe
    wvf_aligned = align_htf_to_ltf(prices, df_12h, wvf)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Calculate 6wk high/low for normalization (using 12h data approximation)
    # 6 weeks = 42 periods of 12h data
    highest_high_42 = np.zeros_like(high_12h)
    lowest_low_42 = np.zeros_like(low_12h)
    
    for i in range(len(high_12h)):
        start_idx = max(0, i - 41)
        highest_high_42[i] = np.max(high_12h[start_idx:i+1])
        lowest_low_42[i] = np.min(low_12h[start_idx:i+1])
    
    # Normalize WVF to 0-100 range using 6wk high/low
    wvf_normalized = np.zeros_like(wvf)
    for i in range(len(wvf)):
        if highest_high_42[i] != lowest_low_42[i]:
            wvf_normalized[i] = ((wvf[i] - 0) / (100 - 0)) * 100  # WVF is already 0-100
        else:
            wvf_normalized[i] = wvf[i]
    
    wvf_norm_aligned = align_htf_to_ltf(prices, df_12h, wvf_normalized)
    
    # Align 12h close for price comparison
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    # Calculate 6h volume moving average for confirmation
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(22, n):
        # Skip if data not available
        if (np.isnan(wvf_norm_aligned[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(vol_ma_6h[i]) or 
            np.isnan(close[i]) or np.isnan(close_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: both 6h and 12h above average
        vol_ok_6h = volume[i] > vol_ma_6h[i]
        vol_ok_12h = df_12h['volume'].values[min(i//2, len(df_12h['volume'].values)-1)] > vol_ma_12h[i//2] if i//2 < len(df_12h) else False
        
        # Trend filter: price above/below 12h EMA
        above_ema = close[i] > ema_12h_aligned[i]
        below_ema = close[i] < ema_12h_aligned[i]
        
        # Williams VIX Fix conditions
        wvf_high = wvf_norm_aligned[i] > 80  # High volatility (fear)
        wvf_low = wvf_norm_aligned[i] < 20   # Low volatility (complacency)
        
        if position == 1:  # Long position
            # Exit: volatility drops (mean reversion complete) or reverse signal
            if wvf_low or (wvf_norm_aligned[i] < wvf_norm_aligned[i-1] and below_ema):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: volatility drops or reverse signal
            if wvf_low or (wvf_norm_aligned[i] < wvf_norm_aligned[i-1] and above_ema):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok_6h and vol_ok_12h:
                # Fade high volatility spikes - mean reversion
                if wvf_high and above_ema:
                    # High volatility during uptrend - expect pullback, go short
                    position = -1
                    signals[i] = -0.25
                elif wvf_high and below_ema:
                    # High volatility during downtrend - expect bounce, go long
                    position = 1
                    signals[i] = 0.25
    
    return signals