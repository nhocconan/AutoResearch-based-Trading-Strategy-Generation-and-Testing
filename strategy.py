#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Supertrend + 1d ADX Trend Filter + Volume Confirmation
# Hypothesis: Supertrend on 6h captures trend direction and momentum. 
# ADX on 1d filters for trending vs ranging markets - we only trade when ADX > 25.
# Volume confirmation ensures institutional participation. Works in both bull and bear markets
# by following the trend direction. 6h timeframe reduces noise vs lower timeframes.
# Target: 15-40 trades/year (60-160 over 4 years).
name = "6h_supertrend_1d_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Supertrend parameters (60-period ATR, multiplier 3.0)
    atr_period = 60
    multiplier = 3.0
    
    # Calculate ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.zeros(n)
    uptrend = np.ones(n, dtype=bool)
    
    for i in range(1, n):
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
    
    # ADX calculation on 1d data (14-period)
    period_adx = 14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]),
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], tr_1d])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_sum = pd.Series(tr_1d).ewm(alpha=1/period_adx, adjust=False, min_periods=period_adx).mean().values
    dm_plus_sum = pd.Series(dm_plus).ewm(alpha=1/period_adx, adjust=False, min_periods=period_adx).mean().values
    dm_minus_sum = pd.Series(dm_minus).ewm(alpha=1/period_adx, adjust=False, min_periods=period_adx).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_sum / tr_sum
    di_minus = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/period_adx, adjust=False, min_periods=period_adx).mean().values
    
    # Align 1d ADX to 6h
    adx_6h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(supertrend[i]) or np.isnan(adx_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check for trending market (ADX > 25)
        trending = adx_6h[i] > 25
        
        if position == 1:  # Long position
            # Exit: Supertrend flips to downtrend OR ADX drops below 20 (ranging)
            if not uptrend[i] or adx_6h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: Supertrend flips to uptrend OR ADX drops below 20 (ranging)
            if uptrend[i] or adx_6h[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require trending market and volume confirmation
            if trending and vol_filter[i]:
                # Enter long when Supertrend is uptrend
                if uptrend[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short when Supertrend is downtrend
                elif not uptrend[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals