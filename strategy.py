#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14111_6d_camarilla1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla levels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    H = high_1d
    L = low_1d
    C = close_1d
    R4 = C + ((H - L) * 1.1 / 2)
    R3 = C + ((H - L) * 1.1 / 4)
    S3 = C - ((H - L) * 1.1 / 4)
    S4 = C - ((H - L) * 1.1 / 2)
    
    # Align to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for volume, 14 for ATR)
    start = max(20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(s4_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Camarilla-based signals with volume confirmation
        # Fade at S3/R3: Long at S3 bounce, Short at R3 rejection
        # Breakout at S4/R4: Break below S4 for short, break above R4 for long
        fade_long = (close[i] > s3_aligned[i-1]) and (close[i-1] <= s3_aligned[i-1]) and vol_filter[i]
        fade_short = (close[i] < r3_aligned[i-1]) and (close[i-1] >= r3_aligned[i-1]) and vol_filter[i]
        breakout_long = (close[i] > r4_aligned[i-1]) and vol_filter[i]
        breakout_short = (close[i] < s4_aligned[i-1]) and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if fade_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif fade_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            elif breakout_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif breakout_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or fade signal at R3
            if close[i] <= stop_price or fade_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or fade signal at S3
            if close[i] >= stop_price or fade_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals