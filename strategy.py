#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
- Long when price breaks above Donchian upper band AND close > 1d EMA34 AND volume > 1.5x 20-period average
- Short when price breaks below Donchian lower band AND close < 1d EMA34 AND volume > 1.5x 20-period average
- Exit when price crosses Donchian middle band (mean reversion to center)
- Uses 1d EMA34 for HTF trend alignment to avoid counter-trend entries
- Volume confirmation reduces false breakouts
- Designed for both bull and bear markets: trend filter prevents counter-trend entries
- Target: 19-50 trades/year (75-200 total over 4 years) to minimize fee drag
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
    
    # Get 1d data for EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Donchian channels from previous day (using 1d data)
    # Upper band = 20-period high, Lower band = 20-period low, Middle band = (upper+lower)/2
    lookback = 20
    if len(df_1d) < lookback + 1:
        return np.zeros(n)
    
    dc_upper = pd.Series(df_1d['high']).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    dc_lower = pd.Series(df_1d['low']).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    dc_middle = (dc_upper + dc_lower) / 2
    
    # Align Donchian levels to 4h timeframe
    dc_upper_aligned = align_htf_to_ltf(prices, df_1d, dc_upper)
    dc_lower_aligned = align_htf_to_ltf(prices, df_1d, dc_lower)
    dc_middle_aligned = align_htf_to_ltf(prices, df_1d, dc_middle)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback + 1, 34, 20)  # Need lookback+1 for Donchian, 34 for EMA, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(dc_upper_aligned[i]) or 
            np.isnan(dc_lower_aligned[i]) or 
            np.isnan(dc_middle_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > dc_upper_aligned[i]  # Break above upper band
        breakout_down = close[i] < dc_lower_aligned[i]  # Break below lower band
        
        # Trend filter (using 1d EMA34)
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Donchian breakout up + uptrend + volume confirmation
            if breakout_up and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + downtrend + volume confirmation
            elif breakout_down and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses Donchian middle band (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below middle band
                if close[i] < dc_middle_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Price crosses above middle band
                if close[i] > dc_middle_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0