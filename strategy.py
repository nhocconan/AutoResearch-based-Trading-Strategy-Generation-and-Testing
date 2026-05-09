#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Supertrend_With_ATR_Stop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ATR for Supertrend (using 1d data for longer-term volatility)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR calculation
    atr_period = 10
    atr = np.zeros_like(tr)
    atr[atr_period] = np.mean(tr[1:atr_period+1])
    for i in range(atr_period + 1, len(tr)):
        atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Supertrend calculation
    multiplier = 3.0
    upper_band = (high_1d + low_1d) / 2 + multiplier * atr
    lower_band = (high_1d + low_1d) / 2 - multiplier * atr
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_1d)
    direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > supertrend[i-1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            supertrend[i] = upper_band[i]
            direction[i] = -1
            
        # Adjust bands
        if direction[i] == 1:
            if lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            supertrend[i] = lower_band[i]
        else:
            if upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
            supertrend[i] = upper_band[i]
    
    # Align Supertrend and direction to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    # Calculate volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        st = supertrend_aligned[i]
        dir_signal = direction_aligned[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: Supertrend uptrend + price above Supertrend + volume confirmation
            if dir_signal == 1 and close[i] > st and vol > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Supertrend downtrend + price below Supertrend + volume confirmation
            elif dir_signal == -1 and close[i] < st and vol > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Supertrend turns down OR price falls below Supertrend
            if dir_signal == -1 or close[i] < st:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Supertrend turns up OR price rises above Supertrend
            if dir_signal == 1 or close[i] > st:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals