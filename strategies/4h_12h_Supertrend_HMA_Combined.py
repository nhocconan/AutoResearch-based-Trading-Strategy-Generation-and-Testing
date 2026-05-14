#!/usr/bin/env python3
# 4h_12h_Supertrend_HMA_Combined
# Hypothesis: Combines 4h Supertrend (trend direction) with 12h HMA (momentum) to filter entries.
# Takes long when both indicate uptrend, short when both indicate downtrend.
# Uses volume confirmation to avoid false breakouts and ATR-based stop loss for risk management.
# Designed for 4h timeframe with 12h as higher timeframe filter to reduce whipsaw.
# Target: 20-40 trades/year per symbol with disciplined risk management for both bull and bear markets.

name = "4h_12h_Supertrend_HMA_Combined"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate ATR for Supertrend
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full_like(tr, np.nan)
    if len(tr) >= 10:
        atr[9] = np.mean(tr[0:10])  # SMA seed for ATR
        for i in range(10, len(tr)):
            atr[i] = (atr[i-1] * 9 + tr[i]) / 10
    
    # Supertrend calculation (4h)
    period = 10
    multiplier = 3.0
    hl2 = (high + low) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    supertrend = np.full_like(close, np.nan)
    uptrend = np.full_like(close, True)
    
    for i in range(1, len(close)):
        if np.isnan(upperband[i-1]) or np.isnan(lowerband[i-1]) or np.isnan(atr[i]):
            supertrend[i] = np.nan
            uptrend[i] = uptrend[i-1] if i > 0 else True
            continue
            
        if close[i] > upperband[i-1]:
            uptrend[i] = True
        elif close[i] < lowerband[i-1]:
            uptrend[i] = False
        else:
            uptrend[i] = uptrend[i-1]
            if uptrend[i] and lowerband[i] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            if not uptrend[i] and upperband[i] > upperband[i-1]:
                upperband[i] = upperband[i-1]
        
        supertrend[i] = lowerband[i] if uptrend[i] else upperband[i]
    
    # Get 12h data for HMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 16:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate Hull Moving Average (16-period)
    def calculate_hma(arr, period):
        n = len(arr)
        hma = np.full(n, np.nan)
        if n < period:
            return hma
        
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA function
        def wma(values, window):
            wma_vals = np.full(len(values), np.nan)
            if len(values) < window:
                return wma_vals
            weights = np.arange(1, window + 1)
            for i in range(window-1, len(values)):
                wma_vals[i] = np.dot(values[i-window+1:i+1], weights) / weights.sum()
            return wma_vals
        
        wma_half = wma(arr, half_period)
        wma_full = wma(arr, period)
        
        # Calculate 2*WMA(half) - WMA(full)
        raw_hma = 2 * wma_half - wma_full
        
        # Final WMA of raw_hma with sqrt_period
        hma = wma(raw_hma, sqrt_period)
        return hma
    
    hma_12h = calculate_hma(close_12h, 16)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)  # Need volume MA and enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend[i]) or np.isnan(hma_12h_aligned[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend direction from both timeframes
        supertrend_up = close[i] > supertrend[i]
        hma_up = close[i] > hma_12h_aligned[i]
        
        if position == 0:
            # Enter long: both timeframes uptrend + volume confirmation
            if supertrend_up and hma_up and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: both timeframes downtrend + volume confirmation
            elif not supertrend_up and not hma_up and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: either timeframe turns downtrend or ATR-based stop
            if not supertrend_up or not hma_up or close[i] < supertrend[i] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: either timeframe turns uptrend or ATR-based stop
            if supertrend_up or hma_up or close[i] > supertrend[i] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals