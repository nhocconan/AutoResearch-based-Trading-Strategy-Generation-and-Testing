#!/usr/bin/env python3
"""
Experiment #7627: 6h Williams Alligator + Elder Ray with 1d trend filter.
Hypothesis: In bull markets (price > 1d EMA50), go long when Alligator jaws < teeth < lips and Elder Bull Power > 0.
In bear markets (price < 1d EMA50), go short when Alligator jaws > teeth > lips and Elder Bear Power < 0.
Uses Williams Alligator (SMAs of median price) for trend and Elder Ray for momentum confirmation.
Targets 80-180 trades over 4 years (20-45/year) with strict alignment conditions.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7627_6h_alligator_elder_1d_ema_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD_JAWS = 13  # Smoothed SMA
ALLIGATOR_PERIOD_TEETH = 8
ALLIGATOR_PERIOD_LIPS = 5
EMA_TREND = 50
ELDER_PERIOD = 13
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d_50 = pd.Series(close_1d).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator: SMAs of median price
    median_price = (high + low) / 2
    jaws = pd.Series(median_price).rolling(window=ALLIGATOR_PERIOD_JAWS, min_periods=ALLIGATOR_PERIOD_JAWS).mean().values
    teeth = pd.Series(median_price).rolling(window=ALLIGATOR_PERIOD_TEETH, min_periods=ALLIGATOR_PERIOD_TEETH).mean().values
    lips = pd.Series(median_price).rolling(window=ALLIGATOR_PERIOD_LIPS, min_periods=ALLIGATOR_PERIOD_LIPS).mean().values
    
    # Smoothed with additional periods (Williams method)
    jaws = pd.Series(jaws).rolling(window=ALLIGATOR_PERIOD_TEETH, min_periods=ALLIGATOR_PERIOD_TEETH).mean().values
    teeth = pd.Series(teeth).rolling(window=ALLIGATOR_PERIOD_LIPS, min_periods=ALLIGATOR_PERIOD_LIPS).mean().values
    
    # Elder Ray Power
    ema_close = pd.Series(close).ewm(span=ELDER_PERIOD, adjust=False, min_periods=ELDER_PERIOD).mean().values
    bull_power = high - ema_close
    bear_power = low - ema_close
    
    # ATR for stoploss
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
    start = max(ALLIGATOR_PERIOD_JAWS, ALLIGATOR_PERIOD_TEETH, ALLIGATOR_PERIOD_LIPS, EMA_TREND, ELDER_PERIOD, ATR_PERIOD) + 5
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1d_50_aligned[i]):
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
        
        # Determine market regime from 1d EMA50
        bull_regime = close[i] > ema_1d_50_aligned[i]   # price above 1d EMA50
        bear_regime = close[i] < ema_1d_50_aligned[i]   # price below 1d EMA50
        
        # Alligator alignment conditions
        # Bullish alignment: jaws < teeth < lips (all lines rising)
        bull_alligator = (jaws[i] < teeth[i]) and (teeth[i] < lips[i])
        # Bearish alignment: jaws > teeth > lips (all lines falling)
        bear_alligator = (jaws[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # Elder Ray conditions
        bull_elder = bull_power[i] > 0
        bear_elder = bear_power[i] < 0
        
        # Entry conditions
        long_entry = bull_regime and bull_alligator and bull_elder
        short_entry = bear_regime and bear_alligator and bear_elder
        
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