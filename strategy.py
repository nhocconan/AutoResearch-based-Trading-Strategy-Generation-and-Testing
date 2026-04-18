#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R combined with 1d ADX trend filter and volume confirmation.
In trending markets (ADX > 25): Williams %R overbought/oversold signals indicate continuation.
In ranging markets (ADX < 20): Williams %R extremes signal mean reversion.
Volume confirmation ensures institutional participation. Designed for 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R."""
    if len(close) < period:
        return np.full(len(close), np.nan)
    
    highest_high = np.full(len(close), np.nan)
    lowest_low = np.full(len(close), np.nan)
    
    for i in range(period-1, len(close)):
        highest_high[i] = np.max(high[i-(period-1):i+1])
        lowest_low[i] = np.min(low[i-(period-1):i+1])
    
    williams_r = np.full(len(close), np.nan)
    for i in range(period-1, len(close)):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
        else:
            williams_r[i] = -50
    
    return williams_r

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range."""
    if len(close) < period:
        return np.full(len(close), np.nan)
    
    tr = np.full(len(close), np.nan)
    for i in range(1, len(close)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = np.full(len(close), np.nan)
    atr[period-1] = np.mean(tr[1:period])
    for i in range(period, len(close)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr

def calculate_dmi(high, low, close, period=14):
    """Calculate Directional Movement Indicators (ADX components)."""
    if len(close) < period + 1:
        return np.full(len(close), np.nan), np.full(len(close), np.nan), np.full(len(close), np.nan)
    
    # Calculate True Range and Directional Movement
    tr = np.full(len(close), np.nan)
    dm_plus = np.full(len(close), np.nan)
    dm_minus = np.full(len(close), np.nan)
    
    for i in range(1, len(close)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            dm_plus[i] = up_move
        else:
            dm_plus[i] = 0
            
        if down_move > up_move and down_move > 0:
            dm_minus[i] = down_move
        else:
            dm_minus[i] = 0
    
    # Smooth TR, DM+, DM-
    atr = np.full(len(close), np.nan)
    atr[period] = np.sum(tr[1:period+1]) / period
    dm_plus_smooth = np.full(len(close), np.nan)
    dm_minus_smooth = np.full(len(close), np.nan)
    dm_plus_smooth[period] = np.sum(dm_plus[1:period+1]) / period
    dm_minus_smooth[period] = np.sum(dm_minus[1:period+1]) / period
    
    for i in range(period+1, len(close)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # Calculate DI+ and DI-
    di_plus = np.full(len(close), np.nan)
    di_minus = np.full(len(close), np.nan)
    dx = np.full(len(close), np.nan)
    
    for i in range(period, len(close)):
        if atr[i] != 0:
            di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
            if di_plus[i] + di_minus[i] != 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
            else:
                dx[i] = 0
        else:
            di_plus[i] = 0
            di_minus[i] = 0
            dx[i] = 0
    
    # Calculate ADX
    adx = np.full(len(close), np.nan)
    adx[2*period] = np.mean(dx[period:2*period+1])
    for i in range(2*period+1, len(close)):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx, di_plus, di_minus

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d
    adx_1d, _, _ = calculate_dmi(high_1d, low_1d, close_1d, 14)
    
    # Calculate Williams %R on 12h (use the same data but calculate directly)
    williams_r = calculate_williams_r(high, low, close, 14)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align ADX to 12h timeframe
    adx_1d_12h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need Williams %R, ADX, and volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(adx_1d_12h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # In trending market (ADX > 25): Williams %R signals continuation
            # In ranging market (ADX < 20): Williams %R signals mean reversion
            if adx_1d_12h[i] > 25:
                # Trending: oversold in uptrend, overbought in downtrend
                if williams_r[i] < -80 and vol_confirmed:  # Oversold
                    signals[i] = 0.25
                    position = 1
                elif williams_r[i] > -20 and vol_confirmed:  # Overbought
                    signals[i] = -0.25
                    position = -1
            else:
                # Ranging: extreme readings mean revert
                if williams_r[i] < -80 and vol_confirmed:  # Oversold
                    signals[i] = 0.25
                    position = 1
                elif williams_r[i] > -20 and vol_confirmed:  # Overbought
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns from oversold or ADX weakens
            if williams_r[i] > -50 or adx_1d_12h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns from overbought or ADX weakens
            if williams_r[i] < -50 or adx_1d_12h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_ADX_Volume"
timeframe = "12h"
leverage = 1.0