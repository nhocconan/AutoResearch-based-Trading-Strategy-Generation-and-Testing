#!/usr/bin/env python3
"""
6H SuperTrend with 12H RSI Filter and Volume Confirmation
Long when price is above SuperTrend AND 12H RSI > 50 AND volume above average
Short when price is below SuperTrend AND 12H RSI < 50 AND volume above average
Exit when price crosses SuperTrend in opposite direction
Designed to capture trends while filtering noise with RSI and volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_supertrend_rsi_12h_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === SuperTrend (10, 3.0) ===
    atr_period = 10
    atr_multiplier = 3.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl_avg = (high + low) / 2
    upper_band = hl_avg + (atr_multiplier * atr)
    lower_band = hl_avg - (atr_multiplier * atr)
    
    # Final Bands
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    final_upper[0] = upper_band[0]
    final_lower[0] = lower_band[0]
    
    for i in range(1, n):
        if close[i-1] <= final_upper[i-1]:
            final_upper[i] = min(upper_band[i], final_upper[i-1])
        else:
            final_upper[i] = upper_band[i]
            
        if close[i-1] >= final_lower[i-1]:
            final_lower[i] = max(lower_band[i], final_lower[i-1])
        else:
            final_lower[i] = lower_band[i]
    
    # SuperTrend
    supertrend = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    supertrend[0] = final_lower[0]
    direction[0] = 1
    
    for i in range(1, n):
        if close[i] > final_upper[i-1]:
            direction[i] = 1
        elif close[i] < final_lower[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = final_lower[i]
        else:
            supertrend[i] = final_upper[i]
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)  # Avoid division by zero
    
    # === 12H RSI filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # RSI (14)
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(supertrend[i]) or np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below SuperTrend
            if close[i] < supertrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above SuperTrend
            if close[i] > supertrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if vol_ratio[i] < 1.1:
                signals[i] = 0.0
                continue
            
            # Entry conditions
            if (close[i] > supertrend[i] and 
                rsi_12h_aligned[i] > 50 and 
                rsi_12h_aligned[i] > rsi_12h_aligned[i-1]):
                # Price above SuperTrend, RSI > 50 and rising -> long
                position = 1
                signals[i] = 0.25
            elif (close[i] < supertrend[i] and 
                  rsi_12h_aligned[i] < 50 and 
                  rsi_12h_aligned[i] < rsi_12h_aligned[i-1]):
                # Price below SuperTrend, RSI < 50 and falling -> short
                position = -1
                signals[i] = -0.25
    
    return signals