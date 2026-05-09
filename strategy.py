#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1-week EMA trend filter and volume confirmation
# Uses weekly high/low/close to calculate Camarilla levels R3/S3. 
# Long when: close > R3, weekly EMA(50) rising, volume spike (>1.5x 20-period average)
# Short when: close < S3, weekly EMA(50) falling, volume spike
# Exit when: price crosses the midpoint of R3/S3 (Camarilla pivot) OR trend reverses
# Position size: 0.25 (25% of capital) to limit drawdown. Target: 12-37 trades/year.
# Designed to work in both bull (breakouts) and bear (mean-reversion at extremes) markets.

name = "12h_Camarilla_R3S3_1wTrend_VolumeSpike"
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
    
    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous week's high, low, close
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels
    camarilla_r3 = close_1w + ((high_1w - low_1w) * 1.2500)
    camarilla_s3 = close_1w - ((high_1w - low_1w) * 1.2500)
    camarilla_pivot = (camarilla_r3 + camarilla_s3) / 2  # Midpoint for exit
    
    # Align weekly Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    
    # Get 1-week data for trend filter
    close_1w = df_1w['close']
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_prev = np.roll(ema_50_1w, 1)
    ema_50_1w_prev[0] = ema_50_1w[0]
    ema_rising = ema_50_1w > ema_50_1w_prev
    ema_falling = ema_50_1w < ema_50_1w_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_falling)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_rising_aligned[i]) or
            np.isnan(ema_falling_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > Camarilla R3 + weekly EMA rising + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < Camarilla S3 + weekly EMA falling + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot OR trend turns down
            if (close[i] < camarilla_pivot_aligned[i]) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot OR trend turns up
            if (close[i] > camarilla_pivot_aligned[i]) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals