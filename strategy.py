#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with volume confirmation and ATR-based stoploss.
# Enter long when price breaks above 1d Donchian upper channel with volume > 2.0x average.
# Enter short when price breaks below 1d Donchian lower channel with volume > 2.0x average.
# Exit when price closes below/above ATR-based trailing stop from extreme.
# Uses Donchian structure for breakouts, volume for confirmation, and ATR for risk management.
# Works in bull markets (breakouts continue up) and bear markets (breakdowns continue down).
# Uses discrete position sizing (0.25) to control risk. Target: 50-150 total trades over 4 years.

name = "12h_Donchian20_VolumeBreakout_ATRStop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channel calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian(20) channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper and lower channels (20-period)
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Calculate 12h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_stop = 0.0
    short_stop = 0.0
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > upper_aligned[i]
        short_breakout = close[i] < lower_aligned[i]
        
        # Update trailing stops
        if position == 1:
            long_stop = max(long_stop, high[i] - 2.5 * atr[i])
        elif position == -1:
            short_stop = min(short_stop, low[i] + 2.5 * atr[i])
        elif position == 0:
            long_stop = 0.0
            short_stop = 0.0
        
        # Exit conditions: ATR trailing stop hit
        long_exit = position == 1 and close[i] < long_stop
        short_exit = position == -1 and close[i] > short_stop
        
        # Entry conditions
        long_entry = long_breakout and vol_confirm and position <= 0
        short_entry = short_breakout and vol_confirm and position >= 0
        
        # Handle entries and exits
        if long_entry:
            signals[i] = 0.25
            position = 1
            long_stop = high[i] - 2.5 * atr[i]  # Initialize stop
        elif short_entry:
            signals[i] = -0.25
            position = -1
            short_stop = low[i] + 2.5 * atr[i]  # Initialize stop
        elif long_exit or short_exit:
            signals[i] = 0.0
            position = 0
            long_stop = 0.0
            short_stop = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals