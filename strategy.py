#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume confirmation + ATR stoploss
# Long when price breaks above 12h Donchian upper (20-period high) AND 1d volume > 1.5x average
# Short when price breaks below 12h Donchian lower (20-period low) AND 1d volume > 1.5x average
# Exit when price crosses Donchian midline (10-period average of high/low) or stoploss hit
# Uses 12h timeframe for low trade frequency, targets 50-150 total trades over 4 years
# Volume confirmation ensures breakouts have conviction, reducing false signals
# Works in both bull/bear markets by capturing strong directional moves

name = "12h_donchian_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on 12h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # 1d volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    daily_volume = df_1d['volume'].values
    volume_ma = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_threshold)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(volume_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Update stoploss
        if position == 1:  # long position
            stop_price = entry_price - 2.5 * atr[i]
            if close[i] <= stop_price or close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            stop_price = entry_price + 2.5 * atr[i]
            if close[i] >= stop_price or close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with volume confirmation
            # Long: price breaks above Donchian upper + volume confirmation
            if close[i] > donchian_upper[i] and volume[i] > volume_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian lower + volume confirmation
            elif close[i] < donchian_lower[i] and volume[i] > volume_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals