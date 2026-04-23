#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R(14) extreme reversal with 1d EMA50 trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND price > 1d EMA50 AND volume > 1.3x 20-period average.
Short when Williams %R > -20 (overbought) AND price < 1d EMA50 AND volume > 1.3x 20-period average.
Exit when Williams %R crosses back through -50 (mean reversion in momentum).
Williams %R identifies exhaustion points; EMA50 filters trend direction; volume confirms participation.
Designed for 6h timeframe to capture swing reversals in both bull and bear markets with moderate trade frequency.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams %R(14) on 6h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    rr = highest_high - lowest_low
    rr[rr == 0] = 1e-10
    willr = -100 * (highest_high - close) / rr
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 20)  # EMA50 needs 50, Williams %R needs 14, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(willr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        ema50_val = ema50_1d_aligned[i]
        willr_val = willr[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND uptrend (price > EMA50) AND volume confirmation
            if willr_val < -80.0 and price > ema50_val and volume[i] > 1.3 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND downtrend (price < EMA50) AND volume confirmation
            elif willr_val > -20.0 and price < ema50_val and volume[i] > 1.3 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when Williams %R crosses back through -50 (mean reversion)
            if position == 1 and willr_val > -50.0:
                signals[i] = 0.0
                position = 0
            elif position == -1 and willr_val < -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_14_1dEMA50_Trend_VolumeConfirmation_WR50Exit"
timeframe = "6h"
leverage = 1.0