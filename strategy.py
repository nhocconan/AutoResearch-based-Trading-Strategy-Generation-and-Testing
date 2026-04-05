#!/usr/bin/env python3
"""
Experiment #10499: 6h Williams Alligator + Elder Ray + 12h ADX Trend Filter
Hypothesis: Williams Alligator identifies trend direction (jaws/teeth/lips alignment),
Elder Ray measures bull/bear power behind the move, and 12h ADX filters for strong trends.
This combination works in both bull (strong uptrends) and bear (strong downtrends) markets
by only taking trades when ADX > 25 indicates a trending environment. Target: 75-175 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10499_6h_williams_alligator_elder_ray_12h_adx_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD_JAWS = 13  # Smoothed SMA
ALLIGATOR_PERIOD_TEETH = 8
ALLIGATOR_PERIOD_LIPS = 5
ELDER_RAY_PERIOD = 13
ADX_PERIOD = 14
ADX_THRESHOLD = 25
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_smoothed_ma(data, period):
    """Calculate smoothed moving average (SMMA)"""
    sma = pd.Series(data).rolling(window=period, min_periods=period).mean()
    # SMMA: (prev_smma * (period-1) + current_price) / period
    smma = np.full_like(data, np.nan, dtype=float)
    if len(data) >= period:
        smma[period-1] = sma.iloc[period-1]
        for i in range(period, len(data)):
            if not np.isnan(smma[i-1]):
                smma[i] = (smma[i-1] * (period-1) + data[i]) / period
            else:
                smma[i] = np.nan
    return smma

def calculate_adx(high, low, close, period):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Directional Movement
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, PlusDM, MinusDM
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    return adx

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
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h ADX for trend strength filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, ADX_PERIOD)
    
    # Align 12h ADX to 6h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator (using SMMA)
    jaws = calculate_smoothed_ma(close, ALLIGATOR_PERIOD_JAWS)
    teeth = calculate_smoothed_ma(close, ALLIGATOR_PERIOD_TEETH)
    lips = calculate_smoothed_ma(close, ALLIGATOR_PERIOD_LIPS)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13 = pd.Series(close).ewm(span=ELDER_RAY_PERIOD, adjust=False, min_periods=ELDER_RAY_PERIOD).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_PERIOD_JAWS, ELDER_RAY_PERIOD, ADX_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if ADX not available
        if np.isnan(adx_12h_aligned[i]):
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
        
        # Trend filter: 12h ADX > 25 indicates trending market
        strong_trend = adx_12h_aligned[i] > ADX_THRESHOLD
        
        # Alligator alignment: jaws < teeth < lips = downtrend, jaws > teeth > lips = uptrend
        alligator_uptrend = (jaws[i] > teeth[i]) and (teeth[i] > lips[i])
        alligator_downtrend = (jaws[i] < teeth[i]) and (teeth[i] < lips[i])
        
        # Elder Ray: bull power > 0 and rising, bear power > 0 and rising
        bull_power_positive = bull_power[i] > 0
        bear_power_positive = bear_power[i] > 0
        bull_power_rising = i > 0 and bull_power[i] > bull_power[i-1]
        bear_power_rising = i > 0 and bear_power[i] > bear_power[i-1]
        
        # Entry conditions
        long_entry = strong_trend and alligator_uptrend and bull_power_positive and bull_power_rising
        short_entry = strong_trend and alligator_downtrend and bear_power_positive and bear_power_rising
        
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