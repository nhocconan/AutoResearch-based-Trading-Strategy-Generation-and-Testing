#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Williams %R (14-period) for mean reversion ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate highest high and lowest low over 14 days
    highest_high = np.full_like(high_1d, np.nan)
    lowest_low = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 13:
            highest_high[i] = np.max(high_1d[i-13:i+1])
            lowest_low[i] = np.min(low_1d[i-13:i+1])
        else:
            highest_high[i] = np.max(high_1d[:i+1]) if i > 0 else high_1d[0]
            lowest_low[i] = np.min(low_1d[:i+1]) if i > 0 else low_1d[0]
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close_1d[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # === 1d ATR (14-period) for volatility filter ===
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Align indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # === 4h Close for price action ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(close_4h[i // 16] if i // 16 < len(close_4h) else np.nan)):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get 4h close price (using last completed 4h bar)
        idx_4h = i // 16
        if idx_4h >= len(close_4h):
            idx_4h = len(close_4h) - 1
        close_4h_price = close_4h[idx_4h]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Williams %R oversold (< -80) + volatility filter
            if (williams_r_aligned[i] < -80 and 
                atr_14_aligned[i] > 0.005 * close[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Williams %R overbought (> -20) + volatility filter
            elif (williams_r_aligned[i] > -20 and 
                  atr_14_aligned[i] > 0.005 * close[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or reverse signal
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or reverse signal
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_MeanReversion_VolatilityFilter_v1"
timeframe = "4h"
leverage = 1.0