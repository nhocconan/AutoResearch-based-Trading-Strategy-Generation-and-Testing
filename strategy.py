#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Supertrend_1w_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Supertrend parameters
    atr_period = 10
    atr_mult = 3.0
    
    # Calculate ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate upper and lower bands
    hl2 = (high + low) / 2
    upperband = hl2 + atr_mult * atr
    lowerband = hl2 - atr_mult * atr
    
    # Initialize Supertrend
    supertrend = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(atr_period, n):
        if i == atr_period:
            supertrend[i] = lowerband[i]
            direction[i] = 1
        else:
            if supertrend[i-1] == upperband[i-1]:
                if close[i] <= upperband[i]:
                    supertrend[i] = upperband[i]
                else:
                    supertrend[i] = lowerband[i]
                    direction[i] = -1
            else:
                if close[i] >= lowerband[i]:
                    supertrend[i] = lowerband[i]
                else:
                    supertrend[i] = upperband[i]
                    direction[i] = 1
    
    # Weekly trend: EMA34 on 1w
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 1.5x 20-period SMA
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period, 50)
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(supertrend[i]) or np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Supertrend uptrend, weekly uptrend, volume confirmation
            if (direction[i] == 1 and 
                price > ema34_1w_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: Supertrend downtrend, weekly downtrend, volume confirmation
            elif (direction[i] == -1 and 
                  price < ema34_1w_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: Supertrend turns down or weekly trend fails
            if (direction[i] == -1 or 
                price < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Supertrend turns up or weekly trend fails
            if (direction[i] == 1 or 
                price > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals