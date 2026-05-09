#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator with 1w trend filter and volume confirmation
# Uses 6h price to calculate Alligator (Jaw, Teeth, Lips). 
# Long when: Lips > Teeth > Jaw (bullish alignment), 1w EMA(8) rising, volume spike (>1.5x 20-period average)
# Short when: Lips < Teeth < Jaw (bearish alignment), 1w EMA(8) falling, volume spike
# Exit when: Alligator lines cross (Lips crosses Teeth) OR trend reverses
# Position size: 0.25 (25% of capital) to limit drawdown. Target: 12-37 trades/year on 6h.
# Williams Alligator catches trends early; 1w filter ensures alignment with higher timeframe trend.
# Works in both bull (trend following) and bear (trend continuation) markets.

name = "6h_WilliamsAlligator_1wTrend_VolumeSpike"
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
    
    # Williams Alligator on 6h data
    # Jaw (13-period SMMA, 8 bars ahead)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # shift 8 bars forward
    
    # Teeth (8-period SMMA, 5 bars ahead)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # shift 5 bars forward
    
    # Lips (5-period SMMA, 3 bars ahead)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # shift 3 bars forward
    
    jaw = jaw.values
    teeth = teeth.values
    lips = lips.values
    
    # Bullish/bearish alignment
    bullish_alignment = (lips > teeth) & (teeth > jaw)
    bearish_alignment = (lips < teeth) & (teeth < jaw)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 8:
        return np.zeros(n)
    
    # 1w EMA(8) for trend filter
    close_1w = df_1w['close']
    ema_8_1w = close_1w.ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_8_1w_prev = np.roll(ema_8_1w, 1)
    ema_8_1w_prev[0] = ema_8_1w[0]
    ema_rising = ema_8_1w > ema_8_1w_prev
    ema_falling = ema_8_1w < ema_8_1w_prev
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
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish alignment + 1w EMA rising + volume spike
            if bullish_alignment[i] and ema_rising_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment + 1w EMA falling + volume spike
            elif bearish_alignment[i] and ema_falling_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Lips crosses below Teeth OR trend turns down
            if (lips[i] < teeth[i]) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Lips crosses above Teeth OR trend turns up
            if (lips[i] > teeth[i]) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals