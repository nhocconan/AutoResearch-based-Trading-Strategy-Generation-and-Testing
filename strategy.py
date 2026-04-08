#!/usr/bin/env python3
"""
6h_1d_rsi_extreme_v1
Hypothesis: Use daily RSI extremes (RSI > 70 or < 30) to identify overbought/oversold conditions on 1d timeframe.
Enter short when RSI > 70 and price breaks below 6h VWAP; enter long when RSI < 30 and price breaks above 6h VWAP.
Requires volume confirmation to avoid false signals. Works in both bull and bear markets by fading extremes.
Target: 15-30 trades/year per symbol (60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_rsi_extreme_v1"
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
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    alpha = 1.0 / 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[:13] = np.nan  # Not enough data
    
    # Align RSI to 6h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 6h VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Volume confirmation: volume > 1.5x average of last 24 periods (4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(vwap[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below VWAP or RSI returns to neutral
            if close[i] < vwap[i] or (rsi_1d_aligned[i] > 40 and rsi_1d_aligned[i] < 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price crosses above VWAP or RSI returns to neutral
            if close[i] > vwap[i] or (rsi_1d_aligned[i] > 40 and rsi_1d_aligned[i] < 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: RSI oversold (< 30) and price crosses above VWAP with volume
            if (rsi_1d_aligned[i] < 30 and 
                close[i] > vwap[i] and 
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: RSI overbought (> 70) and price crosses below VWAP with volume
            elif (rsi_1d_aligned[i] > 70 and 
                  close[i] < vwap[i] and 
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals