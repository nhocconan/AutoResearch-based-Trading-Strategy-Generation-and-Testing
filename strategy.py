#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Supertrend_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR for Supertrend (10-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation (10, 3.0)
    hl2 = (high + low) / 2
    upper_band = hl2 + 3.0 * atr
    lower_band = hl2 - 3.0 * atr
    
    # Initialize Supertrend arrays
    supertrend = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    # Calculate Supertrend
    for i in range(10, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            continue
            
        # Upper and lower bands
        upper = upper_band[i]
        lower = lower_band[i]
        
        # Previous values
        prev_supertrend = supertrend[i-1]
        prev_direction = direction[i-1]
        
        # Current close
        curr_close = close[i]
        
        # Update bands based on direction
        if prev_direction == 1:
            upper = min(upper, upper_band[i-1]) if not np.isnan(upper_band[i-1]) else upper
        else:
            lower = max(lower, lower_band[i-1]) if not np.isnan(lower_band[i-1]) else lower
        
        # Determine trend direction
        if curr_close > upper:
            direction[i] = 1
        elif curr_close < lower:
            direction[i] = -1
        else:
            direction[i] = prev_direction
            if direction[i] == 1 and curr_close < upper:
                upper = upper_band[i]
            elif direction[i] == -1 and curr_close > lower:
                lower = lower_band[i]
        
        # Set Supertrend value
        supertrend[i] = lower if direction[i] == 1 else upper
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(supertrend[i]) or np.isnan(direction[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Supertrend uptrend, price above Supertrend, 12h uptrend, volume spike
            long_cond = (direction[i] == 1 and 
                        close[i] > supertrend[i] and 
                        ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and
                        volume_spike[i])
            
            # Short: Supertrend downtrend, price below Supertrend, 12h downtrend, volume spike
            short_cond = (direction[i] == -1 and 
                         close[i] < supertrend[i] and 
                         ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Supertrend flips to downtrend OR price crosses below Supertrend
            if direction[i] == -1 or close[i] < supertrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Supertrend flips to uptrend OR price crosses above Supertrend
            if direction[i] == 1 or close[i] > supertrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals