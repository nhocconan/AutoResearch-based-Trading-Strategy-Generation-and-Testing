#!/usr/bin/env python3
"""
Experiment #7799: 6-hour Williams Alligator + Elder Ray with 12-hour trend filter.
Hypothesis: The Williams Alligator (Jaw/Teeth/Lips) identifies trending vs ranging markets,
while Elder Ray (Bull Power/Bear Power) measures trend strength. Combined with 12h EMA trend filter,
this captures sustained moves in both bull and bear markets while avoiding whipsaw in ranges.
Targets 50-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7799_6h_alligator_elder_ray_12h_ema_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW = 13    # Smoothed SMA
ALLIGATOR_TEETH = 8   # Smoothed SMA
ALLIGATOR_LIPS = 5    # Smoothed SMA
ELDER_RAY_POWER = 13  # EMA for power calculation
EMA_TREND = 50        # 12h EMA for trend filter
VOLUME_MA = 20        # Volume moving average
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def smma(series, period):
    """Smoothed Moving Average (SMMA)"""
    sma = np.full_like(series, np.nan, dtype=float)
    if len(series) >= period:
        sma[period-1] = np.mean(series[:period])
        for i in range(period, len(series)):
            sma[i] = (sma[i-1] * (period-1) + series[i]) / period
    return sma

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator (using SMMA)
    jaw = smma(high, ALLIGATOR_JAW)  # Typically uses median price, but high works for trend
    teeth = smma(high, ALLIGATOR_TEETH)
    lips = smma(high, ALLIGATOR_LIPS)
    
    # Elder Ray Power
    ema_power = pd.Series(close).ewm(span=ELDER_RAY_POWER, adjust=False, min_periods=ELDER_RAY_POWER).mean().values
    bull_power = high - ema_power
    bear_power = low - ema_power
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA, min_periods=VOLUME_MA).mean().values
    
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
    
    # Start from warmup period
    start = max(ALLIGATOR_JAW, ALLIGATOR_TEETH, ALLIGATOR_LIPS, ELDER_RAY_POWER, EMA_TREND, VOLUME_MA, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_12h_aligned[i]):
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
        
        # Williams Alligator: check if aligned (trending) or tangled (ranging)
        # Alligator aligned when lips > teeth > jaw (bullish) or lips < teeth < jaw (bearish)
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        
        # Skip if any Alligator line is not available
        if np.isnan(lips_val) or np.isnan(teeth_val) or np.isnan(jaw_val):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        bullish_aligned = lips_val > teeth_val > jaw_val
        bearish_aligned = lips_val < teeth_val < jaw_val
        
        # Elder Ray: power confirmation
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Trend filter from 12h EMA
        bull_trend = close[i] > ema_12h_aligned[i]
        bear_trend = close[i] < ema_12h_aligned[i]
        
        # Entry conditions
        # Long: bullish Alligator alignment + bullish Elder Ray + bullish 12h trend + volume
        long_entry = (bullish_aligned and 
                     bull_power_val > 0 and 
                     bull_trend and 
                     volume_confirmed)
        
        # Short: bearish Alligator alignment + bearish Elder Ray + bearish 12h trend + volume
        short_entry = (bearish_aligned and 
                      bear_power_val < 0 and 
                      bear_trend and 
                      volume_confirmed)
        
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