#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + volume confirmation + ATR-based trailing stop.
Long when price breaks above 20-period high with volume > 1.5x average, short when breaks below 20-period low.
Exit on ATR trailing stop (3x ATR from extreme) or opposite Donchian breakout.
Uses discrete position sizing (0.25) to minimize fee churn. Designed for 4h timeframe to target 75-200 trades over 4 years.
Works in bull markets via trend continuation and bear markets via mean reversion at extremes with volume confirmation.
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
    
    # Calculate ATR for trailing stop
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = np.zeros_like(close)
        if len(tr) > period:
            atr[period] = np.mean(tr[1:period+1])
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # Calculate Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    long_stop = 0.0
    short_stop = 0.0
    
    start_idx = 40  # warmup for Donchian (20) + ATR (14) + volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_conf = volume_confirm[i]
        atr_val = atr[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume confirmation
            if price > upper and vol_conf:
                signals[i] = 0.25
                position = 1
                long_stop = price - 3.0 * atr_val  # initial stop
            # Short: price breaks below lower Donchian with volume confirmation
            elif price < lower and vol_conf:
                signals[i] = -0.25
                position = -1
                short_stop = price + 3.0 * atr_val  # initial stop
        
        elif position == 1:
            # Update trailing stop for long position
            long_stop = max(long_stop, price - 3.0 * atr_val)
            # Exit long: price hits stop or breaks below lower Donchian (opposite signal)
            if price <= long_stop or price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update trailing stop for short position
            short_stop = min(short_stop, price + 3.0 * atr_val)
            # Exit short: price hits stop or breaks above upper Donchian (opposite signal)
            if price >= short_stop or price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeConfirm_ATRTrail"
timeframe = "4h"
leverage = 1.0