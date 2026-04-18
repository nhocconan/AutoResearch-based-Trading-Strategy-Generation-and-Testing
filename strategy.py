#!/usr/bin/env python3
"""
Hypothesis: 12h strategy combining Camarilla pivot (R1/S1) breakout with 1d ADX(14) trend filter and volume confirmation.
Pivot breakouts capture institutional levels, ADX filters trending markets, volume ensures conviction.
Designed for 12-37 trades/year (50-150 total) to minimize fee drag while capturing high-probability breakouts.
Works in bull markets (buy R1 breakout in uptrend) and bear markets (sell S1 breakout in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate pivot points (using previous day's OHLC)
    # We'll calculate daily pivots from 1d data, then align to 12h
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points: P = (H+L+C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align pivot levels to 12h timeframe (these are valid for the entire day)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate ADX(14) on 1d for trend strength
    # ADX requires +DI, -DI, DX calculations
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # First value is NaN
        
        # Directional Movement
        plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (similar to EMA but with 1/period)
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[1:period])  # Skip first NaN
            for i in range(period, len(data)):
                if not np.isnan(data[i]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
                else:
                    result[i] = result[i-1]
            return result
        
        atr = wilders_smoothing(tr, period)
        plus_di = 100 * wilders_smoothing(plus_dm, period) / atr
        minus_di = 100 * wilders_smoothing(minus_dm, period) / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilders_smoothing(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_12h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate volume moving average (20-period on 12h)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # need volume MA and enough data for ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(adx_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_12h[i] > 25
        
        if position == 0:
            # Long entry: close breaks above R1 with volume and trend
            if (close[i] > r1_12h[i] and 
                vol_confirmed and 
                trending):
                signals[i] = 0.25
                position = 1
            # Short entry: close breaks below S1 with volume and trend
            elif (close[i] < s1_12h[i] and 
                  vol_confirmed and 
                  trending):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: close below pivot or reverse signal
            if close[i] < pivot_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above pivot or reverse signal
            if close[i] > pivot_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_1dADX25_Volume"
timeframe = "12h"
leverage = 1.0