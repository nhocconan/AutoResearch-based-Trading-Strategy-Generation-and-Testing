#!/usr/bin/env python3
"""
4h_200MA_RangeReversal_RangeFilter
Hypothesis: In BTC/ETH, price often reverts to the 200-period moving average after deviating beyond 1.5x ATR.
This strategy enters long when price is below MA200 by >1.5x ATR and short when above by >1.5x ATR,
only in ranging markets (ADX < 25) to avoid trend-following losses. Exits when price crosses MA200.
Designed for low frequency and robustness in both bull and bear markets by fading extremes in ranges.
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate MA200
    ma200 = np.full(n, np.nan)
    if n >= 200:
        ma200[199] = np.mean(close[0:200])
        for i in range(200, n):
            ma200[i] = (ma200[i-1] * 199 + close[i]) / 200
    
    # Calculate ATR(14)
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr[13] = np.mean(tr[0:14])
        for i in range(14, n):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate ADX(14) for ranging filter
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smooth DM and TR
    plus_dm_smooth = np.full(n, np.nan)
    minus_dm_smooth = np.full(n, np.nan)
    tr_smooth = np.full(n, np.nan)
    
    if n >= 14:
        plus_dm_smooth[13] = np.sum(plus_dm[1:14])
        minus_dm_smooth[13] = np.sum(minus_dm[1:14])
        tr_smooth[13] = np.sum(tr[1:14])
        
        for i in range(14, n):
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / 14) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / 14) + minus_dm[i]
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / 14) + tr[i]
    
    # Calculate DI and DX
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    
    if n >= 14:
        for i in range(14, n):
            if tr_smooth[i] != 0:
                plus_di[i] = 100 * (plus_dm_smooth[i] / tr_smooth[i])
                minus_di[i] = 100 * (minus_dm_smooth[i] / tr_smooth[i])
                if (plus_di[i] + minus_di[i]) != 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # Calculate ADX (smoothed DX)
    adx = np.full(n, np.nan)
    if n >= 27:  # 14 + 13 for smoothing
        adx[26] = np.mean(dx[14:27])
        for i in range(27, n):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 27)
    
    for i in range(start_idx, n):
        if (np.isnan(ma200[i]) or np.isnan(atr[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in ranging markets (ADX < 25)
        if adx[i] >= 25:
            signals[i] = 0.0
            position = 0
            continue
        
        # Calculate deviation from MA200 in ATR units
        dev_atr = (close[i] - ma200[i]) / atr[i] if atr[i] > 0 else 0
        
        if position == 0:
            # Long: price below MA200 by >1.5x ATR
            if dev_atr < -1.5:
                signals[i] = 0.25
                position = 1
            # Short: price above MA200 by >1.5x ATR
            elif dev_atr > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back above MA200
            if close[i] > ma200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back below MA200
            if close[i] < ma200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_200MA_RangeReversal_RangeFilter"
timeframe = "4h"
leverage = 1.0