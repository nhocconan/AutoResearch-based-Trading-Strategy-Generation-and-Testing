#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elders Force Index (EFI) with 13-period EMA smoothing and 1d EMA50 trend filter.
# EFI = volume * (close - prior close) measures bull/bear power with volume confirmation.
# Uses EFI(13) crossing zero for entries, filtered by 1d EMA50 to follow higher timeframe trend.
# Designed to capture momentum shifts with volume confirmation, working in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.
name = "6h_EFI13_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EFI: volume * (close - prior close)
    price_change = np.diff(close, prepend=close[0])
    efi_raw = volume * price_change
    
    # Smooth EFI with 13-period EMA
    efi = pd.Series(efi_raw).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(efi[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: EFI crosses above zero with 1d EMA50 uptrend
            if efi[i] > 0 and efi[i-1] <= 0 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: EFI crosses below zero with 1d EMA50 downtrend
            elif efi[i] < 0 and efi[i-1] >= 0 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: EFI crosses below zero
            if efi[i] < 0 and efi[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: EFI crosses above zero
            if efi[i] > 0 and efi[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals