#!/usr/bin/env python3
"""
Experiment #8019: 6-hour Williams Alligator + Elder Ray Momentum with 12h trend filter.
Hypothesis: The Williams Alligator identifies trend presence via jaw/teeth/lips alignment,
while Elder Ray (bull/bear power) measures trend strength. Combining both with a 12h
trend filter (price above/below 12h EMA50) filters whipsaws and captures sustained moves
in both bull and bear markets. Uses moderate position sizing to manage drawdowns.
Target: 75-150 total trades over 4 years.
"""

from mtf_data import get_alt_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8019_6h_alligator_elder_12h_ema_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13   # Smoothed with 8-bar offset
ALLIGATOR_TEETH_PERIOD = 8  # Smoothed with 5-bar offset
ALLIGATOR_LIPS_PERIOD = 5   # Smoothed with 3-bar offset
ELDER_POWER_PERIOD = 13     # EMA for Elder Power calculation
EMA_TREND_PERIOD = 50       # 12h EMA for trend filter
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SIGNAL_SIZE = 0.25

def calculate_alligator(close, jaw_period, teeth_period, lips_period):
    """Williams Alligator: SMMA (Smoothed Moving Average) with specific offsets."""
    # Jaw: SMMA(close, 13) offset by 8 bars
    jaw = pd.Series(close).ewm(alpha=1/jaw_period, adjust=False).mean()
    jaw = jaw.shift(8)  # Offset by 8 bars
    
    # Teeth: SMMA(close, 8) offset by 5 bars
    teeth = pd.Series(close).ewm(alpha=1/teeth_period, adjust=False).mean()
    teeth = teeth.shift(5)  # Offset by 5 bars
    
    # Lips: SMMA(close, 5) offset by 3 bars
    lips = pd.Series(close).ewm(alpha=1/lips_period, adjust=False).mean()
    lips = lips.shift(3)  # Offset by 3 bars
    
    return jaw.values, teeth.values, lips.values

def calculate_elder_power(high, low, close, ema_period):
    """Elder Ray: Bull Power = High - EMA(close), Bear Power = Low - EMA(close)"""
    ema_close = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    bull_power = high - ema_close.values
    bear_power = low - ema_close.values
    return bull_power, bear_power, ema_close.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    
    # Price relative to 12h EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_12h > ema_12h, 1, -1)  # 1=bullish, -1=bearish
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_12h, price_vs_ema)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Williams Alligator
    jaw, teeth, lips = calculate_alligator(close, 
                                          ALLIGATOR_JAW_PERIOD, 
                                          ALLIGATOR_TEETH_PERIOD, 
                                          ALLIGATOR_LIPS_PERIOD)
    
    # Elder Ray Power
    bull_power, bear_power, ema_close = calculate_elder_power(high, low, close, ELDER_POWER_PERIOD)
    
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
    
    # Start from warmup period (max of all indicators + offsets)
    start = max(ALLIGATOR_JAW_PERIOD + 8,  # jaw offset
                ALLIGATOR_TEETH_PERIOD + 5,  # teeth offset
                ALLIGATOR_LIPS_PERIOD + 3,   # lips offset
                ELDER_POWER_PERIOD,
                EMA_TREND_PERIOD,
                ATR_PERIOD) + 5
    
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
        
        # Determine trend bias from 12h EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 12h close above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 12h close below EMA50
        
        # Alligator alignment: 
        # Bullish: Lips > Teeth > Jaw (all aligned upward)
        # Bearish: Lips < Teeth < Jaw (all aligned downward)
        bull_alligator = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bear_alligator = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Elder Ray confirmation:
        # Bullish: Bull Power > 0 and increasing
        # Bearish: Bear Power < 0 and decreasing
        bull_elder = (bull_power[i] > 0) and (i > 0 and bull_power[i] > bull_power[i-1])
        bear_elder = (bear_power[i] < 0) and (i > 0 and bear_power[i] < bear_power[i-1])
        
        # Entry conditions
        long_entry = bull_bias and bull_alligator and bull_elder
        short_entry = bear_bias and bear_alligator and bear_elder
        
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