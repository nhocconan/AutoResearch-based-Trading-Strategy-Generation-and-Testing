#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
- Long when price breaks above Donchian upper band AND close > 12h EMA50 AND volume > 1.5x 20-period average
- Short when price breaks below Donchian lower band AND close < 12h EMA50 AND volume > 1.5x 20-period average
- Exit when price crosses Donchian middle band (mean reversion to center)
- Uses 12h EMA50 for HTF trend alignment to avoid counter-trend entries
- Volume spike threshold set to 1.5x to reduce false breakouts and trade frequency
- Designed for both bull and bear markets: trend filter prevents counter-trend entries
- Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag
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
    
    # Get 12h data for EMA50 trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Donchian channels from 4h data (primary timeframe)
    donchian_period = 20
    upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    middle = (upper + lower) / 2
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_period, 50, 20)  # Need 20 for Donchian, 50 for EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(middle[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > upper[i]  # Break above upper band
        breakout_down = close[i] < lower[i]  # Break below lower band
        
        # Trend filter (using 12h EMA50)
        uptrend = close[i] > ema50_12h_aligned[i]
        downtrend = close[i] < ema50_12h_aligned[i]
        
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
                if close[i] < middle[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Price crosses above middle band
                if close[i] > middle[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0