# Solution
#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 1-day EMA trend filter and volume confirmation.
Long when price breaks above Donchian upper band (20-period), 1-day EMA50 rising, and volume > 1.5x 20-period average.
Short when price breaks below Donchian lower band, 1-day EMA50 falling, and volume > 1.5x average.
Exit when price crosses midline (10-period average of high/low) or EMA trend reverses.
Designed for low trade frequency (<50/year) by requiring multiple confirmations.
Works in both bull and bear markets by following daily trend while using 4h Donchian for entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_20
    donchian_lower = low_20
    # Midline for exit: 10-period average of high/low
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).mean().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).mean().values
    midline = (high_10 + low_10) / 2.0
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after enough data for Donchian
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(midline[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper band, EMA50 rising, volume confirmation
            if (close[i] > donchian_upper[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band, EMA50 falling, volume confirmation
            elif (close[i] < donchian_lower[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below midline OR EMA trend turns down
                if (close[i] < midline[i] or 
                    ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above midline OR EMA trend turns up
                if (close[i] > midline[i] or 
                    ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0