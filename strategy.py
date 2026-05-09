#!/usr/bin/env python3
# Hypothesis: 12h Camarilla pivot breakout with 1d EMA trend filter and volume spike
# Long when: price breaks above Camarilla R3 (1d), 1d EMA(34) rising, volume spike (>1.8x 20-period avg)
# Short when: price breaks below Camarilla S3 (1d), 1d EMA(34) falling, volume spike
# Exit when: price crosses Camarilla pivot (central) or trend reverses
# Position size: 0.25 to limit drawdown. Target: 15-30 trades/year per symbol.
# Designed to work in bull (breakouts) and bear (mean-reversion at extremes) via trend filter.

name = "12h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # High, low, close from previous 1d bar
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Use previous day's OHLC for Camarilla calculation (avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla multipliers
    R3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    S3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    pivot = (high_1d + low_1d + close_1d) / 3
    
    # Align 1d Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_prev = np.roll(ema_34_1d, 1)
    ema_34_1d_prev[0] = ema_34_1d[0]
    ema_rising = ema_34_1d > ema_34_1d_prev
    ema_falling = ema_34_1d < ema_34_1d_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling)
    
    # Volume spike: current volume > 1.8x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.8 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(ema_rising_aligned[i]) or
            np.isnan(ema_falling_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > Camarilla R3 + 1d EMA rising + volume spike
            if (close[i] > R3_aligned[i] and 
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < Camarilla S3 + 1d EMA falling + volume spike
            elif (close[i] < S3_aligned[i] and 
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot OR trend turns down
            if (close[i] < pivot_aligned[i]) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot OR trend turns up
            if (close[i] > pivot_aligned[i]) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals