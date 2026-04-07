#!/usr/bin/env python3
"""
12h_adx_trend_donchian_breakout_v1
Hypothesis: On 12-hour timeframe, use ADX(14) to filter trending markets and Donchian(20) breakouts for entry.
Go long when price breaks above Donchian high with ADX > 25 and rising.
Go short when price breaks below Donchian low with ADX > 25 and rising.
Exit when price crosses the Donchian midpoint or ADX falls below 20 (trend weakening).
Designed for 15-25 trades/year to minimize fee decay while capturing strong trends.
Works in both bull and bear markets as ADX identifies strong trends regardless of direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_adx_trend_donchian_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian Channel (20-period) on 12h timeframe
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(30, donchian_period), n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
            
        if position == 1:  # Long position
            # Exit: price returns to Donchian midpoint OR trend weakens (ADX < 20)
            if close[i] <= donchian_mid[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Donchian midpoint OR trend weakens (ADX < 20)
            if close[i] >= donchian_mid[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter when ADX indicates strong trend (ADX > 25 and rising)
            if i > 0 and not np.isnan(adx_aligned[i-1]):
                adx_rising = adx_aligned[i] > adx_aligned[i-1]
                strong_trend = adx_aligned[i] > 25 and adx_rising
                
                if strong_trend:
                    # Long: price breaks above Donchian high
                    if close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1]:
                        position = 1
                        signals[i] = 0.25
                    # Short: price breaks below Donchian low
                    elif close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1]:
                        position = -1
                        signals[i] = -0.25
    
    return signals