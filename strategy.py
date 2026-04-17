#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and ATR-based volatility filter.
Long when price breaks above upper Donchian channel (20-period high) AND close > 1w EMA34 (uptrend) AND ATR(14) < 0.5 * ATR(50) (low volatility regime).
Short when price breaks below lower Donchian channel (20-period low) AND close < 1w EMA34 (downtrend) AND ATR(14) < 0.5 * ATR(50).
Exit when price touches the middle of the Donchian channel (median of 20-period high/low) or opposite band.
Uses 1d for price action and Donchian levels, 1w for trend filter.
Designed to capture breakouts in low-volatility trending markets. Target: 10-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian(20) channels
    # Upper band = 20-period high
    # Lower band = 20-period low
    # Middle band = median of upper and lower
    lookback = 20
    upper = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    middle = (upper + lower) / 2
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate ATR(14) and ATR(50) for volatility filter on 1d
    # ATR = average of true ranges
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align all indicators to 1d timeframe (primary timeframe)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    middle_aligned = align_htf_to_ltf(prices, df_1d, middle)
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    atr50_aligned = align_htf_to_ltf(prices, df_1d, atr50)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or
            np.isnan(middle_aligned[i]) or
            np.isnan(ema34_aligned[i]) or
            np.isnan(atr14_aligned[i]) or
            np.isnan(atr50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR(14) < 0.5 * ATR(50) (low volatility regime)
        vol_filter = atr14_aligned[i] < 0.5 * atr50_aligned[i]
        
        # Breakout conditions
        breakout_upper = close[i] > upper_aligned[i]
        breakout_lower = close[i] < lower_aligned[i]
        
        # Exit conditions: touch middle band or opposite band
        touch_middle = abs(close[i] - middle_aligned[i]) < 0.001 * close[i]  # within 0.1%
        touch_opposite = (position == 1 and close[i] < lower_aligned[i]) or \
                         (position == -1 and close[i] > upper_aligned[i])
        
        if position == 0:
            # Long: break above upper band with vol filter and uptrend (close > EMA34)
            if (breakout_upper and vol_filter and close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with vol filter and downtrend (close < EMA34)
            elif (breakout_lower and vol_filter and close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch middle band or break below lower band
            if (touch_middle or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch middle band or break above upper band
            if (touch_middle or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_VolFilter_EMA34_Trend"
timeframe = "1d"
leverage = 1.0