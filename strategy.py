#!/usr/bin/env python3
"""
6h_MarketFacilitationIndex_Trend_1d
Hypothesis: Use Market Facilitation Index (MFI) to detect momentum strength on 6h timeframe. 
Long when MFI > 0 and price > 1d EMA50 (uptrend), short when MFI < 0 and price < 1d EMA50 (downtrend).
Exit when MFI crosses zero. This captures trending moves while avoiding choppy markets.
MFI calculation: (Close - Open) / (High - Low) * Volume. Values >0 indicate buying pressure, <0 selling pressure.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
Works in both bull (strong uptrends) and bear (strong downtrends) by following momentum.
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
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Calculate Market Facilitation Index (MFI) on 6h data
    # MFI = (Close - Open) / (High - Low) * Volume
    # Avoid division by zero
    hl_range = high - low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)  # small value to prevent div by zero
    mfi = ((close - open_price) / hl_range) * volume
    
    # Get 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Start after EMA warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        mfi_val = mfi[i]
        ema_50_val = ema_50_aligned[i]
        
        if position == 0:
            # Long: positive MFI (buying pressure) AND price above 1d EMA50 (uptrend)
            if mfi_val > 0 and close[i] > ema_50_val:
                signals[i] = size
                position = 1
            # Short: negative MFI (selling pressure) AND price below 1d EMA50 (downtrend)
            elif mfi_val < 0 and close[i] < ema_50_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: MFI turns negative (loss of buying pressure)
            if mfi_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: MFI turns positive (loss of selling pressure)
            if mfi_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_MarketFacilitationIndex_Trend_1d"
timeframe = "6h"
leverage = 1.0