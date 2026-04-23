#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h Supertrend trend filter and volume confirmation
- Long when price breaks above Donchian upper band AND 12h Supertrend is bullish AND volume > 1.5x 20-period average
- Short when price breaks below Donchian lower band AND 12h Supertrend is bearish AND volume > 1.5x 20-period average
- Exit when price crosses Donchian middle band (mean reversion to center)
- Uses 12h Supertrend for HTF trend alignment to avoid counter-trend entries
- Volume confirmation reduces false breakouts
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
    
    # Get 12h data for Supertrend trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Supertrend
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR
    high_low = df_12h['high'] - df_12h['low']
    high_close = np.abs(df_12h['high'] - df_12h['close'].shift(1))
    low_close = np.abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate Supertrend
    hl2 = (df_12h['high'] + df_12h['low']) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    supertrend = np.zeros(len(df_12h))
    direction = np.ones(len(df_12h))  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(df_12h)):
        if close_12h := df_12h['close'].iloc[i]:
            close_12h_val = close_12h
        else:
            close_12h_val = df_12h['close'].values[i]
        
        if supertrend[i-1] == upper_band[i-1]:
            supertrend[i] = lower_band[i] if close_12h_val > upper_band[i] else upper_band[i]
            direction[i] = -1 if close_12h_val > upper_band[i] else 1
        else:
            supertrend[i] = upper_band[i] if close_12h_val < lower_band[i] else lower_band[i]
            direction[i] = 1 if close_12h_val < lower_band[i] else -1
    
    # Simplified Supertrend calculation (avoiding complex state tracking)
    # Upper and lower bands
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize trend
    trend = np.ones(len(df_12h))  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(df_12h)):
        if df_12h['close'].values[i] > upper_band[i-1]:
            trend[i] = 1
        elif df_12h['close'].values[i] < lower_band[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
            if trend[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
            if trend[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
    
    supertrend_values = np.where(trend == 1, lower_band, upper_band)
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend_values)
    uptrend_12h = supertrend_values == lower_band  # Bullish when price above lower band
    downtrend_12h = supertrend_values == upper_band  # Bearish when price below upper band
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h.astype(float))
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h.astype(float))
    
    # Calculate Donchian channels on 4h data
    donchian_period = 20
    dc_upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    dc_lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    dc_middle = (dc_upper + dc_lower) / 2
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_period, 20)  # Need 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or 
            np.isnan(dc_middle[i]) or 
            np.isnan(uptrend_12h_aligned[i]) or 
            np.isnan(downtrend_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > dc_upper[i]  # Break above upper band
        breakout_down = close[i] < dc_lower[i]  # Break below lower band
        
        # Trend filter (using 12h Supertrend)
        uptrend = uptrend_12h_aligned[i] == 1.0
        downtrend = downtrend_12h_aligned[i] == 1.0
        
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
                if close[i] < dc_middle[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Price crosses above middle band
                if close[i] > dc_middle[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_12hSupertrend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0