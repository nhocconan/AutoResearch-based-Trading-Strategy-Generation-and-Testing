#!/usr/bin/env python3
"""
6h_Teir3_Confluence_Strategy
Hypothesis: Combine 6h price action with 1d structure using Confluence of:
1. 6h RSI(14) extreme reversal (RSI < 30 for long, > 70 for short)
2. 1d Supertrend(10,3) for trend filter (only trade in direction of daily trend)
3. 6h volume spike (> 2x 20-period average) for confirmation
This avoids overtrading by requiring multiple timeframe alignment and extreme conditions.
Works in bull/bear by following daily trend. Target: 15-35 trades/year.
"""

name = "6h_Teir3_Confluence_Strategy"
timeframe = "6h"
leverage = 1.0

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
    
    # 1d data for Supertrend trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Supertrend (ATR=10, multiplier=3)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(10)
    atr_period = 10
    atr = np.full_like(tr, np.nan)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.nanmean(tr[1:atr_period+1])  # Skip first NaN
        for i in range(atr_period, len(tr)):
            if not np.isnan(tr[i]) and not np.isnan(atr[i-1]):
                atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Supertrend calculation
    supertrend = np.full_like(close_1d, np.nan)
    direction = np.full_like(close_1d, np.nan)  # 1 for uptrend, -1 for downtrend
    
    if len(close_1d) >= atr_period and not np.isnan(atr[-1]):
        # Initialize
        hl2 = (high_1d + low_1d) / 2
        upperband = hl2 + 3 * atr
        lowerband = hl2 - 3 * atr
        
        for i in range(atr_period, len(close_1d)):
            if np.isnan(upperband[i-1]) or np.isnan(lowerband[i-1]) or np.isnan(close_1d[i]):
                continue
                
            # Upper and lower band logic
            if close_1d[i-1] > upperband[i-1]:
                upperband[i] = upperband[i-1]
            else:
                upperband[i] = hl2[i] + 3 * atr[i]
                
            if close_1d[i-1] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            else:
                lowerband[i] = hl2[i] - 3 * atr[i]
            
            # Supertrend and direction
            if i == atr_period:
                if close_1d[i] > upperband[i]:
                    supertrend[i] = upperband[i]
                    direction[i] = -1
                else:
                    supertrend[i] = lowerband[i]
                    direction[i] = 1
            else:
                if supertrend[i-1] == upperband[i-1]:
                    if close_1d[i] <= upperband[i]:
                        supertrend[i] = lowerband[i]
                        direction[i] = 1
                    else:
                        supertrend[i] = upperband[i]
                        direction[i] = -1
                else:  # supertrend[i-1] == lowerband[i-1]
                    if close_1d[i] >= lowerband[i]:
                        supertrend[i] = lowerband[i]
                        direction[i] = 1
                    else:
                        supertrend[i] = upperband[i]
                        direction[i] = -1
    
    # 6h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    rsi_period = 14
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    
    if len(gain) >= rsi_period:
        avg_gain[rsi_period-1] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period-1] = np.mean(loss[:rsi_period])
        for i in range(rsi_period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 6h volume spike (> 2x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma20[19] = np.mean(volume[:20])
        for i in range(20, len(volume)):
            vol_ma20[i] = (vol_ma20[i-1] * 19 + volume[i]) / 20
    volume_spike = volume > (2 * vol_ma20)
    
    # Align 1d Supertrend direction to 6h
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Wait for indicators
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(direction_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        trend_up = direction_aligned[i] == 1
        trend_down = direction_aligned[i] == -1
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: RSI oversold + uptrend + volume spike
            if rsi_oversold and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought + downtrend + volume spike
            elif rsi_overbought and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI overbought or trend turns down
            if rsi[i] > 70 or direction_aligned[i] != 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI oversold or trend turns up
            if rsi[i] < 30 or direction_aligned[i] != -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals