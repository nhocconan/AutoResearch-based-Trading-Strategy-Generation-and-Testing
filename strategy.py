#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Supertrend_KAMA_Trend_Volume"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Supertrend on 1d (ATR=10, multiplier=3) - robust trend filter
    tr1 = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr2 = np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.inf], np.maximum(tr1, tr2)])
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic ATR calculation for Supertrend
    hl2 = (high + low) / 2
    upperband = hl2 + (3 * atr)
    lowerband = hl2 - (3 * atr)
    
    # Initialize Supertrend
    supertrend = np.full_like(close, np.nan)
    dir = np.ones_like(close, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(10, len(close)):
        if np.isnan(atr[i]) or np.isnan(hl2[i]):
            continue
            
        if close[i] > upperband[i-1]:
            dir[i] = 1
        elif close[i] < lowerband[i-1]:
            dir[i] = -1
        else:
            dir[i] = dir[i-1]
            
        if dir[i] == 1:
            lowerband[i] = max(lowerband[i], lowerband[i-1])
            supertrend[i] = lowerband[i]
        else:
            upperband[i] = min(upperband[i], upperband[i-1])
            supertrend[i] = upperband[i]
    
    # Align 1d Supertrend to 4h
    supertrend_1d = supertrend
    dir_1d = dir
    supertrend_4h = align_htf_to_ltf(prices, df_1d, supertrend_1d)
    dir_4h = align_htf_to_ltf(prices, df_1d, dir_1d.astype(float))
    
    # KAMA on 4h for entry timing (ER=10)
    change = np.abs(np.diff(close, k=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1) if len(close) > 1 else np.array([])
    # Proper ER calculation
    er = np.zeros_like(close)
    for i in range(9, len(close)):
        if i >= 9:
            price_change = np.abs(close[i] - close[i-9])
            volatility_sum = np.sum(np.abs(np.diff(close[i-9:i+1])))
            if volatility_sum > 0:
                er[i] = price_change / volatility_sum
            else:
                er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    sc[0] = 0
    
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume condition: current volume > 2.0 x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_4h[i]) or np.isnan(dir_4h[i]) or 
            np.isnan(kama[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Price above Supertrend (uptrend) AND price > KAMA AND volume spike
            if dir_4h[i] == 1 and close[i] > kama[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price below Supertrend (downtrend) AND price < KAMA AND volume spike
            elif dir_4h[i] == -1 and close[i] < kama[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR price falls below KAMA
            if dir_4h[i] == -1 or close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns up OR price rises above KAMA
            if dir_4h[i] == 1 or close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals