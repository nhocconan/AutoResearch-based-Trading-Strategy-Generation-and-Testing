#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Supertrend_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Need sufficient data
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d: Calculate Supertrend ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ATR calculation
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_period = 10
    atr = np.full_like(close_1d, np.nan)
    for i in range(atr_period, len(close_1d)):
        if i == atr_period:
            atr[i] = np.nanmean(tr[1:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Supertrend calculation
    factor = 3.0
    hl2 = (high_1d + low_1d) / 2
    upperband = hl2 + (factor * atr)
    lowerband = hl2 - (factor * atr)
    
    # Initialize arrays
    supertrend = np.full_like(close_1d, np.nan)
    uptrend = np.full_like(close_1d, True)
    
    for i in range(1, len(close_1d)):
        if np.isnan(atr[i-1]) or np.isnan(close_1d[i-1]):
            supertrend[i] = np.nan
            uptrend[i] = uptrend[i-1] if i > 0 else True
            continue
            
        if close_1d[i] > upperband[i-1]:
            uptrend[i] = True
        elif close_1d[i] < lowerband[i-1]:
            uptrend[i] = False
        else:
            uptrend[i] = uptrend[i-1]
            if uptrend[i] and lowerband[i] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            if not uptrend[i] and upperband[i] > upperband[i-1]:
                upperband[i] = upperband[i-1]
        
        supertrend[i] = lowerband[i] if uptrend[i] else upperband[i]
    
    # Align 1d indicators to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend.astype(float))
    
    # === 4h: Volume ratio (current vs 20-period average) ===
    close = prices['close'].values
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Get values
        close_val = close[i]
        supertrend_val = supertrend_aligned[i]
        uptrend_val = uptrend_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(supertrend_val) or np.isnan(uptrend_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Uptrend on 1d, price above Supertrend, volume confirmation
            if (uptrend_val > 0.5 and  # Uptrend on 1d
                close_val > supertrend_val and   # Price above Supertrend
                vol_ratio_val > 1.5):    # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Downtrend on 1d, price below Supertrend, volume confirmation
            elif (uptrend_val < 0.5 and  # Downtrend on 1d
                  close_val < supertrend_val and   # Price below Supertrend
                  vol_ratio_val > 1.5):    # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops below Supertrend (trend reversal)
            if close_val < supertrend_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above Supertrend (trend reversal)
            if close_val > supertrend_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals