#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R3/S3 Breakout with 1d EMA34 Trend Filter and Volume Spike
# Long when: close > Camarilla R3 (1d high-low range), 1d EMA(34) rising, volume > 1.5x 20-period average
# Short when: close < Camarilla S3 (1d high-low range), 1d EMA(34) falling, volume > 1.5x 20-period average
# Exit when: price crosses Camarilla pivot point (PP) OR trend reverses
# Position size: 0.25 (25% of capital) to limit drawdown. Target: 15-35 trades/year.
# Designed to work in both bull (breakouts) and bear (mean-reversion at extremes) markets.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "6h"
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
    
    # Calculate Camarilla levels from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Use previous day's high-low range for today's Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: based on previous day's range
    # H-L = high_1d[i-1] - low_1d[i-1]
    # R4 = close_1d[i-1] + 1.5 * (high_1d[i-1] - low_1d[i-1])
    # R3 = close_1d[i-1] + 1.125 * (high_1d[i-1] - low_1d[i-1])
    # S3 = close_1d[i-1] - 1.125 * (high_1d[i-1] - low_1d[i-1])
    # PP = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3
    
    hl_range = high_1d - low_1d
    r3 = close_1d + 1.125 * hl_range
    s3 = close_1d - 1.125 * hl_range
    pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Shift by 1 to use previous day's levels (avoid look-ahead)
    r3 = np.roll(r3, 1)
    s3 = np.roll(s3, 1)
    pp = np.roll(pp, 1)
    r3[0] = r3[1] if len(r3) > 1 else 0
    s3[0] = s3[1] if len(s3) > 1 else 0
    pp[0] = pp[1] if len(pp) > 1 else 0
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_prev = np.roll(ema_34_1d, 1)
    ema_34_1d_prev[0] = ema_34_1d[0]
    ema_rising = ema_34_1d > ema_34_1d_prev
    ema_falling = ema_34_1d < ema_34_1d_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > R3 + 1d EMA rising + volume spike
            if (close[i] > r3_aligned[i] and 
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < S3 + 1d EMA falling + volume spike
            elif (close[i] < s3_aligned[i] and 
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below PP OR trend turns down
            if (close[i] < pp_aligned[i]) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above PP OR trend turns up
            if (close[i] > pp_aligned[i]) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals