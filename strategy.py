#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with Elder Ray power and 1d trend filter.
# The Alligator (Jaw/Teeth/Lips) identifies trends via smoothed moving averages.
# Elder Ray measures bull/bear power via EMA(13) deviation.
# Combined with 1d EMA trend filter to avoid counter-trend trades.
# Works in bull/bear because Alligator catches trends, Elder Ray filters fakeouts,
# and 1d EMA ensures alignment with higher timeframe momentum.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost.

name = "exp_13047_6h_alligator_elder_1d_ema_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13  # Smoothed MA (8 periods shift)
ALLIGATOR_TEETH_PERIOD = 8  # Smoothed MA (5 periods shift)
ALLIGATOR_LIPS_PERIOD = 5   # Smoothed MA (3 periods shift)
ELDER_RAY_EMA_PERIOD = 13   # EMA for Elder Ray calculation
EMA_1D_PERIOD = 50          # Daily EMA for trend filter
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def smma(series, period):
    """Smoothed Moving Average (SMMA) - Wilder's smoothing"""
    return pd.Series(series).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, EMA_1D_PERIOD)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator: three SMMA lines
    jaw = smma(close, ALLIGATOR_JAW_PERIOD)  # Blue line (13-period)
    teeth = smma(close, ALLIGATOR_TEETH_PERIOD)  # Red line (8-period)
    lips = smma(close, ALLIGATOR_LIPS_PERIOD)   # Green line (5-period)
    
    # Shift the lines as per Alligator specification
    jaw = np.roll(jaw, ALLIGATOR_JAW_PERIOD // 2)
    teeth = np.roll(teeth, ALLIGATOR_TEETH_PERIOD // 2)
    lips = np.roll(lips, ALLIGATOR_LIPS_PERIOD // 2)
    
    # Elder Ray Power: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
    ema_13 = calculate_ema(close, ELDER_RAY_EMA_PERIOD)
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(
        ALLIGATOR_JAW_PERIOD, ALLIGATOR_TEETH_PERIOD, ALLIGATOR_LIPS_PERIOD,
        ELDER_RAY_EMA_PERIOD, EMA_1D_PERIOD, ATR_PERIOD
    ) + 5
    
    for i in range(start, n):
        # Skip if EMA not available
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
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips[i] > teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] < jaw[i]
        
        # Elder Ray confirmation: strong bull/bear power
        strong_bull = bull_power[i] > 0 and bull_power[i] > np.mean(bull_power[max(0, i-20):i+1])
        strong_bear = bear_power[i] > 0 and bear_power[i] > np.mean(bear_power[max(0, i-20):i+1])
        
        # Trend filter: price above/below daily EMA
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Entry signals
        if position == 0:
            # Long: Alligator aligned up + Elder Ray bull + uptrend on 1d
            if alligator_long and strong_bull and uptrend:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: Alligator aligned down + Elder Ray bear + downtrend on 1d
            elif alligator_short and strong_bear and downtrend:
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