#!/usr/bin/env python3
"""
12h_1w_1d_Momentum_Confluence_v1
Concept: 12h momentum with 1w trend filter and 1d volume confirmation.
- Long: Price > 200-period EMA (1w) AND RSI(14) > 55 (12h) AND volume > 1.5x average (1d)
- Short: Price < 200-period EMA (1w) AND RSI(14) < 45 (12h) AND volume > 1.5x average (1d)
- Exit: RSI crosses back to neutral (45-55 range)
- Position sizing: 0.25
- Target: 15-30 trades/year (60-120 total over 4 years)
- Works in bull/bear: 1w EMA200 defines primary trend, 12h RSI captures momentum, 1d volume confirms conviction
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Momentum_Confluence_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 210:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for 200-period EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1w: EMA200 trend filter ===
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # === 12h: RSI(14) momentum ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1d: Volume confirmation ===
    volume_1d = df_1d['volume'].values
    avg_volume_30 = pd.Series(volume_1d).rolling(window=30, min_periods=30).mean().values
    avg_volume_30_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_30)
    
    volume_current = prices['volume'].values
    volume_ratio = volume_current / avg_volume_30_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        ema200_val = ema200_1w_aligned[i]
        rsi_val = rsi[i]
        vol_ratio = volume_ratio[i]
        
        # Skip if any value is invalid
        if (np.isnan(ema200_val) or np.isnan(rsi_val) or np.isnan(vol_ratio) or 
            vol_ratio <= 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Above weekly EMA200 AND bullish RSI AND volume confirmation
            if close[i] > ema200_val and rsi_val > 55 and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Below weekly EMA200 AND bearish RSI AND volume confirmation
            elif close[i] < ema200_val and rsi_val < 45 and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral or turns bearish
            if rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral or turns bullish
            if rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals