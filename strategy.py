#!/usr/bin/env python3
"""
Experiment #8031: 6-hour Williams Alligator + Elder Ray with 1-day trend filter.
Hypothesis: Combining Williams Alligator (trend detection) with Elder Ray (bull/bear power)
on 6h timeframe, filtered by 1d price position relative to EMA50, captures sustained trends
while avoiding whipsaw in both bull and bear markets. Williams Alligator uses smoothed
medians (Jaw/Teeth/Lips) to filter noise, Elder Ray measures bull/bear power via EMA13.
Target: 50-150 total trades over 4 years (12-37/year).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8031_6h_alligator_elder_1d_ema_v1"
timeframe = "6h"
leverage = 1.0

# Williams Alligator parameters (smoothed medians)
JAW_PERIOD = 13   # Blue line
TEETH_PERIOD = 8  # Red line
LIPS_PERIOD = 5   # Green line
JAW_SHIFT = 8
TEETH_SHIFT = 5
LIPS_SHIFT = 3

# Elder Ray parameters
ELDER_EMA_PERIOD = 13

# Trend filter
TREND_EMA_PERIOD = 50

# Risk management
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

# Signal size
SIGNAL_SIZE = 0.25

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=TREND_EMA_PERIOD, adjust=False, min_periods=TREND_EMA_PERIOD).mean().values
    price_vs_ema = np.where(close_1d > ema_1d, 1, -1)  # 1=bullish bias, -1=bearish bias
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Williams Alligator (using smoothed medians)
    # Jaw: 13-period SMMA shifted 8 bars
    jaw_raw = pd.Series(high + low + close).rolling(window=JAW_PERIOD, min_periods=JAW_PERIOD).mean() / 3
    jaw = jaw_raw.shift(JAW_SHIFT).values
    
    # Teeth: 8-period SMMA shifted 5 bars
    teeth_raw = pd.Series(high + low + close).rolling(window=TEETH_PERIOD, min_periods=TEETH_PERIOD).mean() / 3
    teeth = teeth_raw.shift(TEETH_SHIFT).values
    
    # Lips: 5-period SMMA shifted 3 bars
    lips_raw = pd.Series(high + low + close).rolling(window=LIPS_PERIOD, min_periods=LIPS_PERIOD).mean() / 3
    lips = lips_raw.shift(LIPS_SHIFT).values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13 = pd.Series(close).ewm(span=ELDER_EMA_PERIOD, adjust=False, min_periods=ELDER_EMA_PERIOD).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period - need enough data for all indicators
    start = max(JAW_PERIOD + JAW_SHIFT, TEETH_PERIOD + TEETH_SHIFT, 
                LIPS_PERIOD + LIPS_SHIFT, ELDER_EMA_PERIOD, TREND_EMA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Skip if any Alligator line is NaN (not enough data)
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            continue
        
        # Williams Alligator conditions:
        # Bullish alignment: Lips > Teeth > Jaw (green > red > blue)
        # Bearish alignment: Lips < Teeth < Jaw (green < red < blue)
        bullish_align = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_align = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Elder Ray conditions:
        # Strong bull power AND weakening bear power for long
        # Strong bear power AND weakening bull power for short
        strong_bull = bull_power[i] > 0 and bull_power[i] > bull_power[i-1]
        strong_bear = bear_power[i] > 0 and bear_power[i] > bear_power[i-1]
        
        # Determine market bias from 1d EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d close above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d close below EMA50
        
        # Entry conditions
        long_entry = bullish_align and strong_bull and bull_bias
        short_entry = bearish_align and strong_bear and bear_bias
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals