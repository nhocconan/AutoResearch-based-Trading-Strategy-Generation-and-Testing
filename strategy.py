#!/usr/bin/env python3
"""
12h_supertrend_1d_trend_v1
Hypothesis: On 12-hour timeframe, use Supertrend indicator with 1-day trend filter to capture medium-term trends while avoiding whipsaws. Supertrend adapts to volatility via ATR, providing dynamic support/resistance. The 1-day trend filter ensures we only trade in the direction of the higher timeframe trend, improving win rate in both bull and bear markets. Designed for 50-150 total trades over 4 years (~12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_supertrend_1d_trend_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate Average True Range"""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]  # First TR is just high-low
    atr = np.zeros_like(tr)
    atr[:period] = np.nan
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def supertrend(high, low, close, period=10, multiplier=3):
    """Calculate Supertrend indicator"""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.full_like(close, np.nan, dtype=float)
    direction = np.full_like(close, np.nan, dtype=float)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close)):
        if close[i] > supertrend[i-1]:
            direction[i] = 1
        else:
            direction[i] = -1
            
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            
    return supertrend, direction

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Supertrend
    st_period = 10
    st_multiplier = 3
    supertrend_val, st_direction = supertrend(high, low, close, st_period, st_multiplier)
    
    # 1d trend filter (using EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(st_period, 50, 20), n):
        # Skip if data not available
        if (np.isnan(supertrend_val[i]) or np.isnan(st_direction[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: require volume above average
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: Supertrend turns bearish OR 1d trend turns bearish
            if st_direction[i] == -1 or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Supertrend turns bullish OR 1d trend turns bullish
            if st_direction[i] == 1 or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Bullish entry: Supertrend bullish AND price above 1d EMA50
                if st_direction[i] == 1 and close[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Bearish entry: Supertrend bearish AND price below 1d EMA50
                elif st_direction[i] == -1 and close[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals