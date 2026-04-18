#!/usr/bin/env python3
"""
4h_GapAndGo_Reversal
Hypothesis: Exploits mean-reversion after overnight gaps on 4h chart. When price gaps significantly above/below prior 4h close and shows exhaustion (via RSI divergence or volume divergence), we expect a reversion to the mean. Works in both bull and bear markets as gaps occur during market open/close reactions and often reverse.
"""

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Parameters
    gap_threshold = 0.015  # 1.5% gap
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30
    vol_ma_period = 20
    vol_div_threshold = 1.5  # volume should be 1.5x average for gap confirmation
    
    # Calculate RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if i == rsi_period:
            avg_gain[i] = np.mean(gain[i-rsi_period+1:i+1])
            avg_loss[i] = np.mean(loss[i-rsi_period+1:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume moving average
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    # Calculate gap: current open vs previous 4h close
    gap_pct = np.zeros(n)
    gap_pct[0] = 0
    for i in range(1, n):
        gap_pct[i] = (open_price[i] - close[i-1]) / close[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, rsi_period, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: gap down (negative gap) with exhaustion signs
            if gap_pct[i] < -gap_threshold and rsi[i] < rsi_oversold and volume[i] > vol_ma[i] * vol_div_threshold:
                signals[i] = 0.25
                position = 1
            # Short: gap up (positive gap) with exhaustion signs
            elif gap_pct[i] > gap_threshold and rsi[i] > rsi_overbought and volume[i] > vol_ma[i] * vol_div_threshold:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI reverts to neutral or gap fills
            if rsi[i] > 50 or gap_pct[i] > -gap_threshold/2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI reverts to neutral or gap fills
            if rsi[i] < 50 or gap_pct[i] < gap_threshold/2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_GapAndGo_Reversal"
timeframe = "4h"
leverage = 1.0