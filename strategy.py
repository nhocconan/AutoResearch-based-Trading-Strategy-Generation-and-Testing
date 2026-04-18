#!/usr/bin/env python3
"""
4h_ADX_Supertrend_Breakout
Hypothesis: Capture strong directional moves using Supertrend(10,3) confirmed by ADX(14) > 25 for trend strength. Enter long when price breaks above Supertrend and ADX confirms uptrend; short when price breaks below and ADX confirms downtrend. Exit when price crosses back through Supertrend or ADX < 20 (trend exhaustion). Uses volume > 1.8x 20-period average for confirmation to avoid false breakouts. Designed for 4h timeframe to balance signal frequency and noise reduction, working in both bull and bear markets by following adaptive trend filters.
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
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full_like(close, np.nan)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.mean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.full_like(close, np.nan)
    direction = np.full_like(close, np.nan)  # 1 for uptrend, -1 for downtrend
    
    if not np.isnan(atr[0]):
        upper_band[0] = hl2[0] + multiplier * atr[0]
        lower_band[0] = hl2[0] - multiplier * atr[0]
        supertrend[0] = upper_band[0]
        direction[0] = 1
    
    for i in range(1, n):
        if np.isnan(atr[i]):
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
            continue
            
        # Update bands
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        # Adjust bands based on previous close
        if close[i-1] <= supertrend[i-1]:
            upper_band[i] = min(upper_band[i], upper_band[i-1])
        else:
            lower_band[i] = max(lower_band[i], lower_band[i-1])
        
        # Determine trend direction
        if close[i] > upper_band[i-1]:
            direction[i] = 1
        elif close[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        # Set Supertrend value
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # ADX calculation
    adx_period = 14
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    
    for i in range(1, n):
        plus_dm[i] = max(high[i] - high[i-1], 0) if high[i] - high[i-1] > high[i-1] - low[i] else 0
        minus_dm[i] = max(low[i-1] - low[i], 0) if high[i-1] - low[i] > high[i] - high[i-1] else 0
    
    # Smooth DM and TR
    def smooth_series(data, period):
        smoothed = np.full_like(data, np.nan)
        if len(data) >= period:
            smoothed[period-1] = np.sum(data[:period])
            for i in range(period, len(data)):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + data[i]
        return smoothed
    
    smoothed_plus_dm = smooth_series(plus_dm, adx_period)
    smoothed_minus_dm = smooth_series(minus_dm, adx_period)
    smoothed_tr = smooth_series(tr, adx_period)
    
    # Calculate DI and DX
    plus_di = np.full_like(close, np.nan)
    minus_di = np.full_like(close, np.nan)
    dx = np.full_like(close, np.nan)
    
    for i in range(adx_period, n):
        if not np.isnan(smoothed_tr[i]) and smoothed_tr[i] != 0:
            plus_di[i] = 100 * smoothed_plus_dm[i] / smoothed_tr[i]
            minus_di[i] = 100 * smoothed_minus_dm[i] / smoothed_tr[i]
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # Calculate ADX (smoothed DX)
    adx = np.full_like(close, np.nan)
    if len(dx) >= adx_period:
        valid_dx = dx[~np.isnan(dx)]
        if len(valid_dx) >= adx_period:
            adx[adx_period-1] = np.mean(valid_dx[:adx_period])
            for i in range(adx_period, len(valid_dx)):
                adx[i] = (adx[i-1] * (adx_period - 1) + valid_dx[i]) / adx_period
            # Align ADX values back to full array
            adx_full = np.full_like(close, np.nan)
            valid_indices = np.where(~np.isnan(dx))[0]
            if len(valid_indices) >= adx_period:
                adx_full[valid_indices[adx_period-1]:] = adx[adx_period-1:]
                adx = adx_full
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_period = 20
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period, adx_period, vol_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(supertrend[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: price above Supertrend, ADX > 25 (strong uptrend), volume confirmation
            if close[i] > supertrend[i] and adx[i] > 25 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price below Supertrend, ADX > 25 (strong downtrend), volume confirmation
            elif close[i] < supertrend[i] and adx[i] > 25 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below Supertrend OR ADX < 20 (trend weakening)
            if close[i] < supertrend[i] or adx[i] < 20:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above Supertrend OR ADX < 20 (trend weakening)
            if close[i] > supertrend[i] or adx[i] < 20:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ADX_Supertrend_Breakout"
timeframe = "4h"
leverage = 1.0