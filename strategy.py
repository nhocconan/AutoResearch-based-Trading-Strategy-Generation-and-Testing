#!/usr/bin/env python3
# Hypothesis: 12h Camarilla pivot (R3/S3) breakout with 1d trend filter and volume confirmation
# Long when: close > Camarilla R3 (1d), 1d EMA(34) rising, volume spike (>1.5x 20-period average)
# Short when: close < Camarilla S3 (1d), 1d EMA(34) falling, volume spike
# Exit when: price crosses Camarilla midpoint (P) OR trend reverses
# Position size: 0.25 (25% of capital) to limit drawdown. Target: 12-37 trades/year.
# Designed to work in both bull (breakouts at R3/S3) and bear (mean-reversion at extremes) markets.

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
    high_1d = get_htf_data(prices, '1d')['high'].values
    low_1d = get_htf_data(prices, '1d')['low'].values
    close_1d = get_htf_data(prices, '1d')['close'].values
    
    # Camarilla: H-L range from previous day
    rng = high_1d - low_1d
    camarilla_r3 = close_1d + (rng * 1.1 / 6)
    camarilla_s3 = close_1d - (rng * 1.1 / 6)
    camarilla_p = (high_1d + low_1d + close_1d) / 3
    
    # Align Camarilla levels to 12h timeframe (previous day's levels available at open)
    camarilla_r3_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1d'), camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1d'), camarilla_s3)
    camarilla_p_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1d'), camarilla_p)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    close_1d_series = df_1d['close']
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
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
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_p_aligned[i]) or np.isnan(ema_rising_aligned[i]) or
            np.isnan(ema_falling_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > Camarilla R3 + 1d EMA rising + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < Camarilla S3 + 1d EMA falling + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Camarilla P OR trend turns down
            if (close[i] < camarilla_p_aligned[i]) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Camarilla P OR trend turns up
            if (close[i] > camarilla_p_aligned[i]) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals