#!/usr/bin/env python3
"""
1H RSI(14) Extreme + 4H Supertrend + Volume Spike (1.5x)
Long: RSI < 30 (oversold) + price > 4H Supertrend + volume > 1.5x 20-period MA
Short: RSI > 70 (overbought) + price < 4H Supertrend + volume > 1.5x 20-period MA
Exit: Opposite RSI extreme (RSI > 50 for long exit, RSI < 50 for short exit)
Size: 0.20
Target: 15-30 trades/year per symbol
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
    
    # 1H RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4H Supertrend (ATR=10, mult=3)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).ewm(alpha=1/10, adjust=False, min_periods=10).mean()
    
    # Basic Upper/Lower Bands
    basic_ub = (high_4h + low_4h) / 2 + 3 * atr
    basic_lb = (high_4h + low_4h) / 2 - 3 * atr
    
    # Final Upper/Lower Bands
    final_ub = np.zeros_like(close_4h)
    final_lb = np.zeros_like(close_4h)
    for i in range(len(close_4h)):
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            if basic_ub[i] < final_ub[i-1] or close_4h[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
            
            if basic_lb[i] > final_lb[i-1] or close_4h[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
    
    # Supertrend
    supertrend = np.zeros_like(close_4h)
    for i in range(len(close_4h)):
        if i == 0:
            supertrend[i] = final_ub[i]
        else:
            if supertrend[i-1] == final_ub[i-1] and close_4h[i] <= final_ub[i]:
                supertrend[i] = final_ub[i]
            elif supertrend[i-1] == final_ub[i-1] and close_4h[i] > final_ub[i]:
                supertrend[i] = final_lb[i]
            elif supertrend[i-1] == final_lb[i-1] and close_4h[i] >= final_lb[i]:
                supertrend[i] = final_lb[i]
            elif supertrend[i-1] == final_lb[i-1] and close_4h[i] < final_lb[i]:
                supertrend[i] = final_ub[i]
    
    # Align 4H Supertrend to 1H
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    
    # Volume MA(20)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma[i]
        
        if position == 0:
            # Long: RSI < 30 + price > Supertrend + volume spike
            if rsi[i] < 30 and price > supertrend_aligned[i] and vol > 1.5 * vol_ma:
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 + price < Supertrend + volume spike
            elif rsi[i] > 70 and price < supertrend_aligned[i] and vol > 1.5 * vol_ma:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 50
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI < 50
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1H_RSI14_Extreme_4HSupertrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0