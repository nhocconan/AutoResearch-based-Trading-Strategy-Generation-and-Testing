#!/usr/bin/env python3
"""
Experiment #8799: 6h Williams Alligator + Elder Ray + 12h Trend Filter
Hypothesis: Combines Alligator's trend detection (jaw/teeth/lips) with Elder Ray's bull/bear power
and 12h trend filter to capture sustained trends while avoiding whipsaws. Works in both bull
and bear markets by only taking trades aligned with higher timeframe trend. Targets 50-150
trades over 4 years (12-37/year) with discrete position sizing to minimize fee drag.
"""

from mtf_data import get_afft_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8799_6h_alligator_elder_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD = 13
ELDER_RAY_PERIOD = 13
TREND_EMA_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_alligator(close, period):
    """Williams Alligator: Jaw=SMMA(close,13,8), Teeth=SMMA(close,13,5), Lips=SMMA(close,13,3)"""
    smma13 = pd.Series(close).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    jaw = smma13.shift(8)  # 8 periods ahead
    teeth = smma13.shift(5)  # 5 periods ahead
    lips = smma13.shift(3)  # 3 periods ahead
    return jaw.values, teeth.values, lips.values

def calculate_elder_ray(high, low, close, period):
    """Elder Ray: Bull Power = High - EMA(close), Bear Power = Low - EMA(close)"""
    ema = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean()
    bull_power = high - ema.values
    bear_power = low - ema.values
    return bull_power, bear_power

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=TREND_EMA_PERIOD, adjust=False, min_periods=TREND_EMA_PERIOD).mean().values
    
    # Price relative to 12h EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_12h > ema_12h, 1, 
                     np.where(close_12h < ema_12h, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_12h, price_vs_ema)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator
    jaw, teeth, lips = calculate_alligator(close, ALLIGATOR_PERIOD)
    
    # Elder Ray
    bull_power, bear_power = calculate_elder_ray(high, low, close, ELDER_RAY_PERIOD)
    
    # ATR for risk management
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_PERIOD + 8, ELDER_RAY_PERIOD, TREND_EMA_PERIOD, ATR_PERIOD) + 1
    
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
        
        # Determine market bias from 12h EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 12h price above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 12h price below EMA50
        
        # Alligator conditions: Lips > Teeth > Jaw = bullish alignment
        alligator_bull = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Alligator conditions: Jaw > Teeth > Lips = bearish alignment
        alligator_bear = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        # Elder Ray conditions
        elder_bull = bull_power[i] > 0  # Bullish when bull power positive
        elder_bear = bear_power[i] < 0  # Bearish when bear power negative
        
        # Entry conditions
        long_entry = bull_bias and alligator_bull and elder_bull
        short_entry = bear_bias and alligator_bear and elder_bear
        
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