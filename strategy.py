#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_ewm_crossover_with_volume_filter
# Uses 60-period EWM (exponential weighted moving average) on 6h chart for trend detection.
# Long when fast EWM(12) crosses above slow EWM(60) with volume confirmation (volume > 1.5x 20-period avg).
# Short when fast EWM(12) crosses below slow EWM(60) with volume confirmation.
# Uses 1d trend filter: only take longs when 60-period EWM on 1d is rising, shorts when falling.
# Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.
# Works in trending markets via momentum and avoids choppy periods via volume confirmation.

name = "6h_1d_ewm_crossover_with_volume_filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 60-period EWM on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema60_1d = pd.Series(close_1d).ewm(span=60, adjust=False, min_periods=60).mean().values
    # Rising when current > previous, falling when current < previous
    ema60_rising = ema60_1d[1:] > ema60_1d[:-1]
    ema60_rising = np.concatenate([[False], ema60_rising])  # align with original length
    ema60_falling = ema60_1d[1:] < ema60_1d[:-1]
    ema60_falling = np.concatenate([[False], ema60_falling])  # align with original length
    
    # Align 1d EWM trend to 6h
    ema60_rising_aligned = align_htf_to_ltf(prices, df_1d, ema60_rising)
    ema60_falling_aligned = align_htf_to_ltf(prices, df_1d, ema60_falling)
    
    # Calculate EWM crossovers on 6h
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Crossover signals
    bullish_cross = (ema12 > ema60) & (np.roll(ema12, 1) <= np.roll(ema60, 1))
    bearish_cross = (ema12 < ema60) & (np.roll(ema12, 1) >= np.roll(ema60, 1))
    # Handle first element
    bullish_cross[0] = False
    bearish_cross[0] = False
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(ema60_rising_aligned[i]) or np.isnan(ema60_falling_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions: bullish crossover + volume confirmation + 1d uptrend
        if bullish_cross[i] and vol_confirm[i] and ema60_rising_aligned[i]:
            if position != 1:
                position = 1
                signals[i] = 0.25
            else:
                signals[i] = 0.25  # maintain position
        # Short conditions: bearish crossover + volume confirmation + 1d downtrend
        elif bearish_cross[i] and vol_confirm[i] and ema60_falling_aligned[i]:
            if position != -1:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = -0.25  # maintain position
        # Exit conditions: opposite crossover
        elif position == 1 and bearish_cross[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bullish_cross[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals