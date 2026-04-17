#!/usr/bin/env python3
"""
4h TriStar Doji Reversal + 1d Volume Spike + Trend Filter
Long: TriStar Doji (3 consecutive dojis) + volume > 1.5x 20-period average + price > 1d EMA50
Short: TriStar Doji + volume spike + price < 1d EMA50
Exit: Opposite TriStar pattern or price crosses 1d EMA50
TriStar Doji is a rare but high-probability reversal pattern indicating market indecision and potential turning point.
Designed for low trade frequency (~20-40/year) with high win rate in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume SMA(20) for volume filter
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate doji condition: |close - open| <= 0.1 * (high - low)
    body = np.abs(close - open_price)
    rng = high - low
    # Avoid division by zero
    rng_safe = np.where(rng == 0, 1, rng)
    doji = body <= (0.1 * rng_safe)
    
    # TriStar Doji: 3 consecutive dojis
    tristar = np.zeros(n, dtype=bool)
    for i in range(2, n):
        if doji[i] and doji[i-1] and doji[i-2]:
            tristar[i] = True
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 50)  # need EMA50 and volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_20[i]) or
            np.isnan(tristar[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_20[i]
        ema_50_val = ema_50_1d_aligned[i]
        is_tristar = tristar[i]
        
        if position == 0:
            # Long: TriStar Doji + volume spike + price > 1d EMA50
            if is_tristar and vol > 1.5 * vol_sma_val and price > ema_50_val:
                signals[i] = 0.25
                position = 1
            # Short: TriStar Doji + volume spike + price < 1d EMA50
            elif is_tristar and vol > 1.5 * vol_sma_val and price < ema_50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Opposite TriStar or price < 1d EMA50
            if is_tristar and vol > 1.5 * vol_sma_val and price < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Opposite TriStar or price > 1d EMA50
            if is_tristar and vol > 1.5 * vol_sma_val and price > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TriStarDoji_Reversal_VolumeSpike_1dEMA50"
timeframe = "4h"
leverage = 1.0