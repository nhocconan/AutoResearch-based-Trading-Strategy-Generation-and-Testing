#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian breakout with volume confirmation and ATR-based stoploss.
# Enter long when price breaks above 12h Donchian(20) upper band and volume > 1.5x 20-bar average.
# Enter short when price breaks below 12h Donchian(20) lower band and volume > 1.5x 20-bar average.
# Exit when price crosses back inside the Donchian bands.
# Donchian channels provide clear structure, volume confirms breakout strength, ATR stoploss controls risk.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue).
# Uses discrete position sizing (0.30) to control risk. Target: 75-200 total trades over 4 years.

name = "4h_Donchian20_12h_VolumeBreakout_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian calculation (HTF)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian(20) channels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: 20-period high
    upper_band = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    lower_band = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_band)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_band)
    
    # Calculate ATR for stoploss (using 4h data)
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align length with prices
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate 4h volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        long_breakout = close[i] > upper_aligned[i]
        short_breakout = close[i] < lower_aligned[i]
        
        # Re-entry conditions (price back inside bands)
        long_exit = close[i] < upper_aligned[i]
        short_exit = close[i] > lower_aligned[i]
        
        # ATR-based stoploss conditions
        if position == 1:  # Long position
            stoploss_level = close[i - position_change] - 2.5 * atr[i] if 'position_change' in locals() else close[i] - 2.5 * atr[i]
            stoploss_hit = close[i] < stoploss_level
        elif position == -1:  # Short position
            stoploss_level = close[i - position_change] + 2.5 * atr[i] if 'position_change' in locals() else close[i] + 2.5 * atr[i]
            stoploss_hit = close[i] > stoploss_level
        else:
            stoploss_hit = False
        
        # Entry conditions
        long_entry = long_breakout and volume_confirm[i]
        short_entry = short_breakout and volume_confirm[i]
        
        # Exit conditions
        long_exit_signal = long_exit or stoploss_hit
        short_exit_signal = short_exit or stoploss_hit
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.30
            position = 1
            position_change = i
        elif short_entry and position >= 0:
            signals[i] = -0.30
            position = -1
            position_change = i
        elif (position == 1 and long_exit_signal) or (position == -1 and short_exit_signal):
            signals[i] = 0.0
            position = 0
            position_change = i
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals