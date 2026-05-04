#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Force Index + 12h KAMA trend filter
# Uses 6h Elder Force Index (EFI = Volume * (Close - Prior Close)) smoothed with EMA(13) to measure buying/selling pressure.
# Combines with 12h Kaufman Adaptive Moving Average (KAMA) as trend filter: only trade when price is above/below KAMA.
# EFI provides momentum timing: long when EFI > 0 and rising, short when EFI < 0 and falling.
# KAMA adapts to market noise, reducing whipsaw in ranging markets while catching trends.
# Designed for 12-30 trades/year (~50-120 total over 4 years) to minimize fee drag.
# Works in both bull/bear markets by combining momentum (EFI) with adaptive trend (KAMA).

name = "6h_ElderForceIndex_12hKAMA_Trend"
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
    
    # Get 12h data for KAMA calculation - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h KAMA ( Kaufman Adaptive Moving Average )
    # Efficiency Ratio (ER) = |Change| / Sum|Changes| over 10 periods
    change = np.abs(np.diff(close_12h))
    abs_change = np.abs(change)
    
    # Pad arrays for calculation
    change_padded = np.concatenate([[0], change])
    abs_change_padded = np.concatenate([[0], abs_change])
    
    # Calculate ER over 10 periods
    er = np.zeros_like(close_12h)
    for i in range(10, len(close_12h)):
        net_change = abs(close_12h[i] - close_12h[i-10])
        total_change = np.sum(abs_change_padded[i-9:i+1])
        if total_change > 0:
            er[i] = net_change / total_change
        else:
            er[i] = 0
    
    # Smoothing constants: fastest EMA(2), slowest EMA(30)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_12h)
    kama[:] = np.nan
    kama[29] = close_12h[29]  # Start after 30 periods for min_periods
    
    for i in range(30, len(close_12h)):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
        else:
            kama[i] = close_12h[i]
    
    # Align KAMA to 6h timeframe (wait for completed 12h bar)
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # Calculate 6h Elder Force Index (EFI)
    # EFI = Volume * (Close - Prior Close)
    price_change = np.diff(close, prepend=close[0])
    efi_raw = volume * price_change
    
    # Smooth EFI with EMA(13)
    efi = pd.Series(efi_raw).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(efi[i]) or np.isnan(efi[i-1])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price above 12h KAMA AND EFI positive AND rising
            if (close[i] > kama_aligned[i] and 
                efi[i] > 0 and 
                efi[i] > efi[i-1]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price below 12h KAMA AND EFI negative AND falling
            elif (close[i] < kama_aligned[i] and 
                  efi[i] < 0 and 
                  efi[i] < efi[i-1]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 12h KAMA OR EFI turns negative
            if (close[i] <= kama_aligned[i]) or (efi[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 12h KAMA OR EFI turns positive
            if (close[i] >= kama_aligned[i]) or (efi[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals