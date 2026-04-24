#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1w EMA50 trend filter and 1d volume spike confirmation.
- Uses weekly EMA50 for primary trend direction (bullish if price > EMA50, bearish if price < EMA50)
- Uses 1d volume spike (>2.0x 20-period average) for entry conviction
- Enters long when price breaks above H3 in bullish weekly trend with volume confirmation
- Enters short when price breaks below L3 in bearish weekly trend with volume confirmation
- Exits on retest of opposite Camarilla level (L3 for longs, H3 for shorts)
- Designed for 12-30 trades/year (50-120 total over 4 years) to stay within fee-efficient range
- Combines Camarilla structure with weekly trend filter and daily volume confirmation for BTC/ETH resilience
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
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior weekly OHLC (completed weekly bar)
    high_1w = df_1w['high'].shift(1).values
    low_1w = df_1w['low'].shift(1).values
    close_1w = df_1w['close'].shift(1).values
    
    # Align weekly data to 6h timeframe
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Calculate weekly EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 6h Camarilla levels from prior weekly bar
    camarilla_h3 = close_1w_aligned + 1.1 * (high_1w_aligned - low_1w_aligned) / 4
    camarilla_l3 = close_1w_aligned - 1.1 * (high_1w_aligned - low_1w_aligned) / 4
    
    # Get daily volume for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x daily average)
        volume_confirm = vol_1d_aligned[i] > 2.0 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long: Close > H3 AND price above weekly EMA50 AND volume confirmation
            if close[i] > camarilla_h3[i] and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Close < L3 AND price below weekly EMA50 AND volume confirmation
            elif close[i] < camarilla_l3[i] and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < L3 (retest of opposite level)
            if close[i] < camarilla_l3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > H3 (retest of opposite level)
            if close[i] > camarilla_h3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1wEMA50_1dVolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0