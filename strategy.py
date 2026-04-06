#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray with weekly trend filter
# Works in bull/bear because Alligator identifies trends (jaws/teeth/lips),
# Elder Ray measures bull/bear power, and weekly trend filters countertrend trades.
# Target: 80-150 trades over 4 years (20-38/year) with proper risk control.

name = "exp_12875_6h_alligator_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD_JAW = 13  # Blue line
ALLIGATOR_PERIOD_TEETH = 8  # Red line
ALLIGATOR_PERIOD_LIPS = 5   # Green line
ELDER_RAY_PERIOD = 13
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_alligator(high, low, close, period_jaw, period_teeth, period_lips):
    """Williams Alligator: SMMA of median price"""
    median_price = (high + low) / 2
    
    def smma(series, period):
        # Smoothed Moving Average (similar to Wilder's smoothing)
        sma = np.full_like(series, np.nan)
        if len(series) >= period:
            sma[period-1] = np.mean(series[:period])
            for i in range(period, len(series)):
                sma[i] = (sma[i-1] * (period-1) + series[i]) / period
        return sma
    
    jaw = smma(median_price, period_jaw)
    teeth = smma(median_price, period_teeth)
    lips = smma(median_price, period_lips)
    
    return jaw, teeth, lips

def calculate_elder_ray(high, low, close, period):
    """Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA"""
    ema_close = pd.Series(close).ewm(span=period, adjust=False).mean().values
    bull_power = high - ema_close
    bear_power = low - ema_close
    return bull_power, bear_power, ema_close

def calculate_atr(high, low, close, period):
    """ATR using Wilder's smoothing"""
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
    
    # Load weekly trend data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly trend: price above/below 21-period EMA
    weekly_close = df_weekly['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=21, adjust=False).mean().values
    weekly_uptrend = weekly_close > weekly_ema
    weekly_downtrend = weekly_close < weekly_ema
    
    # Align weekly trend to 6h
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_downtrend.astype(float))
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    jaw, teeth, lips = calculate_alligator(high, low, close, 
                                          ALLIGATOR_PERIOD_JAW, 
                                          ALLIGATOR_PERIOD_TEETH, 
                                          ALLIGATOR_PERIOD_LIPS)
    bull_power, bear_power, ema_close = calculate_elder_ray(high, low, close, ELDER_RAY_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Warmup: need enough data for Alligator (max period) and ATR
    start = max(ALLIGATOR_PERIOD_JAW, ELDER_RAY_PERIOD, ATR_PERIOD) + 5
    
    for i in range(start, n):
        # Skip if Alligator lines not available
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
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
        
        # Alligator conditions: lips > teeth > jaw = uptrend, lips < teeth < jaw = downtrend
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray: bull power > 0 and rising, bear power < 0 and falling
        bull_strong = bull_power[i] > 0 and (i == start or bull_power[i] > bull_power[i-1])
        bear_strong = bear_power[i] < 0 and (i == start or bear_power[i] < bear_power[i-1])
        
        # Weekly trend filter
        weekly_long_ok = weekly_uptrend_aligned[i] > 0.5
        weekly_short_ok = weekly_downtrend_aligned[i] > 0.5
        
        # Generate signals
        if position == 0:
            # Long: Alligator uptrend + bull power strong + weekly uptrend
            if alligator_long and bull_strong and weekly_long_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: Alligator downtrend + bear power strong + weekly downtrend
            elif alligator_short and bear_strong and weekly_short_ok:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Stay in long until stop or reversal
            signals[i] = SIGNAL_SIZE
            # Optional: exit if Alligator reverses
            if not alligator_long:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Stay in short until stop or reversal
            signals[i] = -SIGNAL_SIZE
            # Optional: exit if Alligator reverses
            if not alligator_short:
                signals[i] = 0.0
                position = 0
    
    return signals