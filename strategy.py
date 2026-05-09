#!/usr/bin/env python3
# Hypothesis: 1h Williams %R overbought/oversold with 4h EMA trend filter and volume confirmation
# Long when: Williams %R < -80 (oversold), 4h EMA(21) rising, volume spike (>1.5x 20-period avg)
# Short when: Williams %R > -20 (overbought), 4h EMA(21) falling, volume spike
# Exit when: Williams %R crosses above -50 (long) or below -50 (short) OR trend reverses
# Position size: 0.20 (20% of capital) to limit drawdown. Target: 15-37 trades/year.
# Williams %R identifies reversal points in ranging markets; EMA filter ensures trend alignment.
# Volume spike confirms institutional interest. Works in both bull (buy oversold dips) and bear (sell overbought rallies).

name = "1h_WilliamsR_4hEMA_VolumeSpike"
timeframe = "1h"
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
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # 4h EMA(21) for trend filter
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_prev = np.roll(ema_21_4h, 1)
    ema_21_4h_prev[0] = ema_21_4h[0]
    ema_rising = ema_21_4h > ema_21_4h_prev
    ema_falling = ema_21_4h < ema_21_4h_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_4h, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_4h, ema_falling)
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_rising_aligned[i]) or
            np.isnan(ema_falling_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R < -80 (oversold) + 4h EMA rising + volume spike
            if (williams_r[i] < -80 and 
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: Williams %R > -20 (overbought) + 4h EMA falling + volume spike
            elif (williams_r[i] > -20 and 
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -50 OR trend turns down
            if (williams_r[i] > -50) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50 OR trend turns up
            if (williams_r[i] < -50) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals