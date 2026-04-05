#!/usr/bin/env python3
"""
Experiment #8839: 6h Williams Alligator + Elder Ray + 12h Trend Filter
Hypothesis: Combines Williams Alligator (Jaw/Teeth/Lips) for trend direction with Elder Ray (Bull/Bear Power) for momentum strength, filtered by 12h EMA trend. Works in bull markets via Alligator alignment and in bear markets via Elder Ray divergences. Targets 50-150 trades over 4 years (12-37/year) to avoid fee drag while maintaining statistical validity.
"""

from mtf_data import get_aff_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8839_6h_alligator_elder_ray_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD_JAW = 13
ALLIGATOR_PERIOD_TEETH = 8
ALLIGATOR_PERIOD_LIPS = 5
ELDER_RAY_PERIOD = 13
TREND_PERIOD = 50
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_alligator(close, period_jaw, period_teeth, period_lips):
    """Williams Alligator: SMMA (Smoothed Moving Average)"""
    def smma(series, period):
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean()
        return sma.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    jaw = smma(close, period_jaw)
    teeth = smma(close, period_teeth)
    lips = smma(close, period_lips)
    return jaw.values, teeth.values, lips.values

def calculate_elder_ray(high, low, close, period):
    """Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA"""
    ema = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean()
    bull_power = high - ema.values
    bear_power = low - ema.values
    return bull_power, bear_power

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
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=TREND_PERIOD, adjust=False, min_periods=TREND_PERIOD).mean().values
    
    # Price relative to 12h EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_12h > ema_12h, 1, 
                     np.where(close_12h < ema_12h, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_12h, price_vs_ema)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator
    jaw, teeth, lips = calculate_alligator(close, ALLIGATOR_PERIOD_JAW, ALLIGATOR_PERIOD_TEETH, ALLIGATOR_PERIOD_LIPS)
    
    # Elder Ray
    bull_power, bear_power = calculate_elder_ray(high, low, close, ELDER_RAY_PERIOD)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_PERIOD_JAW, ALLIGATOR_PERIOD_TEETH, ALLIGATOR_PERIOD_LIPS, 
                ELDER_RAY_PERIOD, TREND_PERIOD, ATR_PERIOD) + 1
    
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
        
        # Williams Alligator conditions
        # Lips > Teeth > Jaw = bullish alignment
        # Lips < Teeth < Jaw = bearish alignment
        alligator_bull = (lips[i] > teeth[i] > jaw[i]) if not (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i])) else False
        alligator_bear = (lips[i] < teeth[i] < jaw[i]) if not (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i])) else False
        
        # Elder Ray conditions
        # Strong bull power and weak bear power = bullish momentum
        # Strong bear power and weak bull power = bearish momentum
        elder_bull = (bull_power[i] > 0 and bear_power[i] < 0) if not (np.isnan(bull_power[i]) or np.isnan(bear_power[i])) else False
        elder_bear = (bear_power[i] > 0 and bull_power[i] < 0) if not (np.isnan(bull_power[i]) or np.isnan(bear_power[i])) else False
        
        # Determine market bias from 12h EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 12h price above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 12h price below EMA50
        
        # Entry conditions: Require alignment between Alligator, Elder Ray, and 12h trend
        long_entry = alligator_bull and elder_bull and bull_bias
        short_entry = alligator_bear and elder_bear and bear_bias
        
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