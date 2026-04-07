#!/usr/bin/env python3
"""
1d_rsi_momentum_1w_trend_volume_v1
Hypothesis: RSI(14) momentum on daily timeframe, filtered by weekly trend (EMA20) and volume spike.
Long when RSI > 55, price above weekly EMA20, and volume > 1.5x average.
Short when RSI < 45, price below weekly EMA20, and volume > 1.5x average.
Designed for low trade frequency (~10-20 trades/year) with strong momentum signals to work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_rsi_momentum_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    ema20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14]) if n >= 14 else 0
    avg_loss[13] = np.mean(loss[1:14]) if n >= 14 else 0
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average (spike)
        vol_spike = volume[i] > (vol_ma[i] * 1.5)
        
        # RSI conditions
        rsi_over = rsi[i] > 55
        rsi_under = rsi[i] < 45
        
        # Weekly trend filter
        above_1w_ema20 = close[i] > ema20_1w_aligned[i]
        below_1w_ema20 = close[i] < ema20_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI turns bearish or trend turns bearish
            if rsi[i] < 50 or below_1w_ema20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI turns bullish or trend turns bullish
            if rsi[i] > 50 or above_1w_ema20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: RSI bullish with volume spike and bullish trend
            if rsi_over and vol_spike and above_1w_ema20:
                position = 1
                signals[i] = 0.25
            # Short: RSI bearish with volume spike and bearish trend
            elif rsi_under and vol_spike and below_1w_ema20:
                position = -1
                signals[i] = -0.25
    
    return signals