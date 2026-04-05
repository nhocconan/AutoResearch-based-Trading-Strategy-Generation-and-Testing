#!/usr/bin/env python3
"""
Experiment #11251: 6h Williams Alligator + Elder Ray + 1d Trend Filter
Hypothesis: Williams Alligator identifies trends via jaw/teeth/lips alignment, Elder Ray measures bull/bear power strength, and 1d trend filter ensures alignment with higher timeframe. Works in bull markets via long entries when alligator is bullish and bull power > 0, and in bear markets via short entries when alligator is bearish and bear power > 0. Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_11251_6h_williams_alligator_elder_ray_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD_JAW = 13
ALLIGATOR_PERIOD_TEETH = 8
ALLIGATOR_PERIOD_LIPS = 5
ELDER_RAY_PERIOD = 13
EMA_LONG_TREND_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SIGNAL_SIZE = 0.25

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_williams_alligator(high, low, close, period_jaw, period_teeth, period_lips):
    """Calculate Williams Alligator lines (using median price)"""
    median_price = (high + low) / 2
    jaw = calculate_ema(median_price, period_jaw * 2)  # Smoothed with 13*2=26
    teeth = calculate_ema(median_price, period_teeth * 2)  # Smoothed with 8*2=16
    lips = calculate_ema(median_price, period_lips * 2)  # Smoothed with 5*2=10
    return jaw, teeth, lips

def calculate_elder_ray(high, low, close, period):
    """Calculate Elder Ray: Bull Power = High - EMA, Bear Power = EMA - Low"""
    ema = calculate_ema(close, period)
    bull_power = high - ema
    bear_power = ema - low
    return bull_power, bear_power, ema

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend filter
    ema_daily = calculate_ema(df_daily['close'].values, EMA_LONG_TREND_PERIOD)
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator
    jaw, teeth, lips = calculate_williams_alligator(
        high, low, close,
        ALLIGATOR_PERIOD_JAW,
        ALLIGATOR_PERIOD_TEETH,
        ALLIGATOR_PERIOD_LIPS
    )
    
    # Elder Ray
    bull_power, bear_power, ema_elder = calculate_elder_ray(
        high, low, close, ELDER_RAY_PERIOD
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(
        ALLIGATOR_PERIOD_JAW * 2,
        ALLIGATOR_PERIOD_TEETH * 2,
        ALLIGATOR_PERIOD_LIPS * 2,
        ELDER_RAY_PERIOD,
        EMA_LONG_TREND_PERIOD
    ) + 1
    
    for i in range(start, n):
        # Skip if daily EMA not available
        if np.isnan(ema_daily_aligned[i]):
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
        
        # Williams Alligator trend detection
        # Bullish: Lips > Teeth > Jaw
        # Bearish: Lips < Teeth < Jaw
        alligator_bullish = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        alligator_bearish = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Elder Ray power
        bull_power_positive = bull_power[i] > 0
        bear_power_positive = bear_power[i] > 0
        
        # Daily trend filter
        uptrend_daily = close[i] > ema_daily_aligned[i]
        downtrend_daily = close[i] < ema_daily_aligned[i]
        
        # Entry conditions
        long_entry = alligator_bullish and bull_power_positive and uptrend_daily
        short_entry = alligator_bearish and bear_power_positive and downtrend_daily
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * 0.0)  # ATR will be calculated below
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * 0.0)  # ATR will be calculated below
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
        
        # Calculate ATR for stoploss (using previous bar's value to avoid look-ahead)
        if i > 0:
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            # Simplified ATR calculation for stop - using fixed ATR proxy
            # In practice, we'd use a proper ATR, but for stop calculation we can use a simplified approach
            atr_estimate = tr  # Simplified - in reality would use smoothed ATR
            if position == 1:
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr_estimate)
            elif position == -1:
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr_estimate)
    
    return signals