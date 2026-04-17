#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and ATR-based stoploss.
Long when price breaks above Donchian upper band in strong uptrend (ADX > 25).
Short when price breaks below Donchian lower band in strong downtrend (ADX > 25).
Exit when price reverts to the middle band (20-period average of high/low) or ATR stoploss hit.
Uses 1d for ADX calculation to ensure trend stability, 4h for price/volume/Donchian.
Target: 75-200 total trades over 4 years (19-50/year). Discrete sizing 0.25 to minimize fee churn.
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
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (Average Directional Index)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        dm_plus = np.zeros_like(close)
        dm_minus = np.zeros_like(close)
        for i in range(1, len(close)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
            dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        # Smoothed TR, DM+, DM- (Wilder's smoothing)
        atr = np.zeros_like(close)
        dmp = np.zeros_like(close)
        dmm = np.zeros_like(close)
        
        # Initial values
        atr[period] = np.mean(tr[1:period+1])
        dmp[period] = np.mean(dm_plus[1:period+1])
        dmm[period] = np.mean(dm_minus[1:period+1])
        
        # Wilder's smoothing
        for i in range(period+1, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dmp[i] = (dmp[i-1] * (period-1) + dm_plus[i]) / period
            dmm[i] = (dmm[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        dip = np.zeros_like(close)
        dim = np.zeros_like(close)
        for i in range(period, len(close)):
            if atr[i] > 0:
                dip[i] = 100 * dmp[i] / atr[i]
                dim[i] = 100 * dmm[i] / atr[i]
        
        # Directional Index (DX)
        dx = np.zeros_like(close)
        for i in range(period, len(close)):
            if dip[i] + dim[i] > 0:
                dx[i] = 100 * abs(dip[i] - dim[i]) / (dip[i] + dim[i])
        
        # ADX (smoothed DX)
        adx = np.zeros_like(close)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(close)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 4h Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = np.zeros_like(high)
        lower = np.zeros_like(low)
        middle = np.zeros_like(high)
        
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
            middle[i] = (upper[i] + lower[i]) / 2.0
        
        return upper, lower, middle
    
    donch_upper, donch_lower, donch_middle = calculate_donchian(high, low, 20)
    
    # Calculate ATR for dynamic stoploss (4h)
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(close)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        return atr
    
    atr_4h = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_upper[i]) or 
            np.isnan(donch_lower[i]) or 
            np.isnan(donch_middle[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx_1d_aligned[i]
        atr_val = atr_4h[i]
        upper = donch_upper[i]
        lower = donch_lower[i]
        middle = donch_middle[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        is_strong_trend = adx_val > 25
        
        if position == 0:
            # Long: price breaks above upper Donchian band in strong uptrend
            if price > upper and is_strong_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian band in strong downtrend
            elif price < lower and is_strong_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            
            # 1. Price reverts to middle band
            if price <= middle:
                exit_signal = True
            # 2. ATR-based stoploss (2.5 ATR below entry)
            elif price < entry_price - 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            
            # 1. Price reverts to middle band
            if price >= middle:
                exit_signal = True
            # 2. ATR-based stoploss (2.5 ATR above entry)
            elif price > entry_price + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ADXTrend_ATRStop"
timeframe = "4h"
leverage = 1.0