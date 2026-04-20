#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_Trend
Hypothesis: Price breakout above Donchian(20) upper/lower channel with volume confirmation and 4h ADX trend filter.
Long when price breaks above upper band with volume surge and strong 4h trend; short when breaks below lower band.
Uses 4h timeframe for all calculations to match the proven winning patterns in the DB.
Target: 20-50 trades per year per symbol with position size 0.25 to manage drawdown and reduce fee drag.
Works in both bull and bear markets by requiring trend alignment (ADX > 25) and volume confirmation.
"""

name = "4h_Donchian20_Breakout_Volume_Trend"
timeframe = "4h"
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
    
    # Calculate Donchian Channel (20-period)
    def calculate_donchian(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    upper_band, lower_band = calculate_donchian(high, low, 20)
    
    # Calculate volume moving average (20-period) for confirmation
    def calculate_sma(data, period):
        sma = np.full_like(data, np.nan)
        for i in range(period-1, len(data)):
            sma[i] = np.mean(data[i-period+1:i+1])
        return sma
    
    volume_sma = calculate_sma(volume, 20)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Plus Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Wilder's smoothing
        def wilder_smoothing(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            if len(data) >= period:
                result[period-1] = np.nansum(data[:period]) / period
                for i in range(period, len(data)):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        atr = wilder_smoothing(tr, period)
        plus_di = 100 * wilder_smoothing(plus_dm, period) / atr
        minus_di = 100 * wilder_smoothing(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilder_smoothing(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(volume_sma[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period average
        volume_surge = volume[i] > 1.5 * volume_sma[i]
        
        if position == 0:
            # Long: price breaks above upper band + volume surge + strong 1d trend (ADX > 25)
            if close[i] > upper_band[i] and volume_surge and adx_1d_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band + volume surge + strong 1d trend (ADX > 25)
            elif close[i] < lower_band[i] and volume_surge and adx_1d_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower band OR ADX weak (< 20)
            if close[i] < lower_band[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper band OR ADX weak (< 20)
            if close[i] > upper_band[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals