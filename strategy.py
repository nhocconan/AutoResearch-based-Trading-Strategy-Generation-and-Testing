#!/usr/bin/env python3
"""
Experiment #12211: 6h Williams Alligator + Elder Ray + 1d Trend Filter
Hypothesis: Williams Alligator identifies trend presence (jaws-teeth-lips alignment) and direction,
Elder Ray measures bull/bear power via EMA(13) deviation, and 1d EMA provides higher-timeframe trend bias.
Combined, they filter false signals in chop and capture sustained trends in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12211_6h_williams_alligator_elder_ray_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAWS_PERIOD = 13  # Smoothed with 8-period shift
ALLIGATOR_TEETH_PERIOD = 8  # Smoothed with 5-period shift
ALLIGATOR_LIPS_PERIOD = 5   # Smoothed with 3-period shift
ELDER_RAY_EMA_PERIOD = 13   # EMA for Bull/Bear power calculation
TREND_EMA_PERIOD = 50       # 1d EMA for trend filter
SIGNAL_SIZE = 0.25          # Position size (25% of capital)
ATR_PERIOD = 14             # ATR for stop loss
ATR_STOP_MULTIPLIER = 2.5   # Stop loss multiplier

def calculate_smma(data, period):
    """Calculate Smoothed Moving Average (SMMA) - used in Williams Alligator"""
    sma = np.full_like(data, np.nan, dtype=float)
    if len(data) >= period:
        sma[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            sma[i] = (sma[i-1] * (period-1) + data[i]) / period
    return sma

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    ema_1d = calculate_ema(df_1d['close'].values, TREND_EMA_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator: three SMMA lines
    jaws = calculate_smma(close, ALLIGATOR_JAWS_PERIOD)  # 13-period, shifted 8 bars
    teeth = calculate_smma(close, ALLIGATOR_TEETH_PERIOD)  # 8-period, shifted 5 bars
    lips = calculate_smma(close, ALLIGATOR_LIPS_PERIOD)    # 5-period, shifted 3 bars
    
    # Apply shifts as per Alligator definition
    jaws = np.roll(jaws, ALLIGATOR_JAWS_PERIOD + 5)  # Shift 8 bars forward? Wait, standard is jaws: 13 SMA shifted 8, teeth: 8 SMA shifted 5, lips: 5 SMA shifted 3
    # Actually, Williams Alligator uses SMMA then shifts jaws by 8, teeth by 5, lips by 3
    # Let me correct: calculate SMMA then shift
    jaws_raw = calculate_smma(close, ALLIGATOR_JAWS_PERIOD)
    teeth_raw = calculate_smma(close, ALLIGATOR_TEETH_PERIOD)
    lips_raw = calculate_smma(close, ALLIGATOR_LIPS_PERIOD)
    
    jaws = np.roll(jaws_raw, 8)   # Jaws: 13-period SMMA shifted 8 bars
    teeth = np.roll(teeth_raw, 5) # Teeth: 8-period SMMA shifted 5 bars
    lips = np.roll(lips_raw, 3)   # Lips: 5-period SMMA shifted 3 bars
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
    ema_13 = calculate_ema(close, ELDER_RAY_EMA_PERIOD)
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Warmup: need enough data for all indicators
    start = max(
        ALLIGATOR_JAWS_PERIOD + 8,  # Jaws needs data + shift
        ALLIGATOR_TEETH_PERIOD + 5, # Teeth needs data + shift
        ALLIGATOR_LIPS_PERIOD + 3,  # Lips needs data + shift
        ELDER_RAY_EMA_PERIOD,
        TREND_EMA_PERIOD
    ) + 1
    
    for i in range(start, n):
        # Skip if 1d EMA not available
        if np.isnan(ema_1d_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
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
        
        # Williams Alligator conditions:
        # Trending market: Lips > Teeth > Jaws (uptrend) or Lips < Teeth < Jaws (downtrend)
        # Avoid trading when Alligator is sleeping (all intertwined)
        alligator_long = (lips[i] > teeth[i]) and (teeth[i] > jaws[i])
        alligator_short = (lips[i] < teeth[i]) and (teeth[i] < jaws[i])
        
        # Elder Ray conditions:
        # Strong bull power: Bull Power > 0 and increasing
        # Strong bear power: Bear Power > 0 and increasing
        bull_power_strong = bull_power[i] > 0 and (i == 0 or bull_power[i] > bull_power[i-1])
        bear_power_strong = bear_power[i] > 0 and (i == 0 or bear_power[i] > bear_power[i-1])
        
        # Trend filter (1d EMA)
        uptrend_1d = close[i] > ema_1d_aligned[i]
        downtrend_1d = close[i] < ema_1d_aligned[i]
        
        # Entry conditions
        long_entry = alligator_long and bull_power_strong and uptrend_1d
        short_entry = alligator_short and bear_power_strong and downtrend_1d
        
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
</export>