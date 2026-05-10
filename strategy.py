#!/usr/bin/env python3
# 6h_Williams_VIX_Fix_Reversal
# Hypothesis: Williams VIX Fix identifies volatility spikes that precede mean-reverting moves in crypto.
# High VIX Fix values indicate panic selling (long opportunity) or euphoric buying (short opportunity).
# Combined with RSI extremes and volume confirmation to filter false signals.
# Designed for 6h to capture multi-day reversals with low trade frequency.

name = "6h_Williams_VIX_Fix_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams VIX Fix: measures synthetic VIX from price action
    # VIX Fix = (Highest Close in period - Low) / Highest Close in period * 100
    def williams_vix_fix(high_arr, low_arr, close_arr, period=22):
        highest_close = np.full_like(close_arr, np.nan)
        for i in range(len(close_arr)):
            if i < period - 1:
                highest_close[i] = np.nan
            else:
                highest_close[i] = np.max(close_arr[i-period+1:i+1])
        vix_fix = (highest_close - low_arr) / highest_close * 100
        return vix_fix
    
    # Calculate Williams VIX Fix
    vix_fix = williams_vix_fix(high, low, close, period=22)
    
    # RSI for confirmation
    def rsi(close_arr, period=14):
        delta = np.diff(close_arr, prepend=close_arr[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(close_arr, np.nan)
        avg_loss = np.full_like(close_arr, np.nan)
        
        # Initialize first average
        if len(close_arr) >= period:
            avg_gain[period-1] = np.mean(gain[0:period])
            avg_loss[period-1] = np.mean(loss[0:period])
            
            # Wilder smoothing
            for i in range(period, len(close_arr)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val
    
    rsi_val = rsi(close, period=14)
    
    # Volume moving average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(vix_fix[i]) or np.isnan(rsi_val[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: High VIX Fix (panic selling) + RSI oversold + volume confirmation
            if vix_fix[i] > 80 and rsi_val[i] < 30 and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: Extremely low VIX Fix (complacency) + RSI overbought + volume confirmation
            elif vix_fix[i] < 20 and rsi_val[i] > 70 and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: VIX Fix normalizes or RSI overbought
            if vix_fix[i] < 40 or rsi_val[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: VIX Fix normalizes or RSI oversold
            if vix_fix[i] > 60 or rsi_val[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals