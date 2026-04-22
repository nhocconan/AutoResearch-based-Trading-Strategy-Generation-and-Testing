#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Supertrend(ATR=10, mult=3) with volume confirmation and session filter
# Uses 1d Supertrend for trend direction to avoid counter-trend trades
# Entry when price crosses Supertrend in direction of higher timeframe trend
# Volume > 1.5x 20-period average confirms momentum
# Target: 20-40 trades/year per symbol, works in bull/bear via trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Supertrend calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day ATR(10)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 1-day Supertrend
    upper_band = (high_1d + low_1d) / 2 + 3 * atr_10
    lower_band = (high_1d + low_1d) / 2 - 3 * atr_10
    
    # Initialize Supertrend arrays
    supertrend = np.full_like(close_1d, np.nan)
    direction = np.full_like(close_1d, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Calculate Supertrend iteratively
    for i in range(1, len(close_1d)):
        if np.isnan(atr_10[i-1]) or np.isnan(close_1d[i-1]):
            continue
            
        # Upper and lower band logic
        if close_1d[i-1] <= upper_band[i-1]:
            upper_band[i] = min(upper_band[i], upper_band[i-1])
        else:
            upper_band[i] = upper_band[i]
            
        if close_1d[i-1] >= lower_band[i-1]:
            lower_band[i] = max(lower_band[i], lower_band[i-1])
        else:
            lower_band[i] = lower_band[i]
        
        # Trend direction
        if close_1d[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_1d[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1] if not np.isnan(direction[i-1]) else 1
        
        # Supertrend value
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Volume spike filter (20-period on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 4-hour timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        # Skip if data not ready or outside session
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price crosses above Supertrend + volume spike + uptrend (direction=1)
            if (close[i] > supertrend_aligned[i] and 
                close[i-1] <= supertrend_aligned[i-1] and 
                vol_spike[i] and 
                direction_aligned[i] == 1):
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below Supertrend + volume spike + downtrend (direction=-1)
            elif (close[i] < supertrend_aligned[i] and 
                  close[i-1] >= supertrend_aligned[i-1] and 
                  vol_spike[i] and 
                  direction_aligned[i] == -1):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back through Supertrend
            if position == 1:
                if close[i] < supertrend_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > supertrend_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Supertrend_ATR10_mult3_Volume_Session"
timeframe = "4h"
leverage = 1.0