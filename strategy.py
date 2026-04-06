#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(50) trend filter and volume confirmation
# Long when price breaks above Donchian high + EMA(50) up + volume > 2x average
# Short when price breaks below Donchian low + EMA(50) down + volume > 2x average
# Exit when price crosses EMA(50) or Donchian midpoint reverses
# Uses 4h timeframe targeting 75-200 total trades over 4 years (19-50/year)
# Volume threshold increased to 2x to reduce trades, EMA faster than KAMA for responsiveness

name = "4h_donchian_ema_vol_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # EMA(50) on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # EMA calculation with proper smoothing
    ema_12h = np.zeros_like(close_12h)
    ema_12h[0] = close_12h[0]
    alpha = 2 / (50 + 1)
    for i in range(1, len(close_12h)):
        ema_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_12h[i-1]
    
    # Align EMA to 4h timeframe
    ema_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: volume > 2x 20-period average (stricter threshold)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses EMA or Donchian midpoint
        if position == 1:  # long position
            if close[i] <= ema_aligned[i] or close[i] <= donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= ema_aligned[i] or close[i] >= donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with trend and volume confirmation
            # Bullish breakout: price above Donchian high + EMA up + volume
            if (close[i] > donch_high[i] and 
                ema_aligned[i] > ema_aligned[i-1] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Bearish breakout: price below Donchian low + EMA down + volume
            elif (close[i] < donch_low[i] and 
                  ema_aligned[i] < ema_aligned[i-1] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals