#!/usr/bin/env python3
"""
Experiment #7631: 6h Williams Alligator + 1-day Elder Ray System
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) defines trend direction and strength on 6h.
Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) from 1d confirms institutional bias.
Only take Alligator signals when aligned with 1d Elder Ray to avoid whipsaws.
Uses Williams %R for entry timing within Alligator alignment.
Targets 80-180 trades over 4 years (20-45/year) with strict alignment filters.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7631_6h_alligator_elder_williamsr_v1"
timeframe = "6h"
leverage = 1.0

# Williams Alligator parameters (6h)
ALLIGATOR_JAW_PERIOD = 13   # Smoothed with 8-period shift
ALLIGATOR_TEETH_PERIOD = 8  # Smoothed with 5-period shift
ALLIGATOR_LIPS_PERIOD = 5   # Smoothed with 3-period shift

# Elder Ray parameters (1d)
ELDER_EMA_PERIOD = 13

# Williams %R for entry timing (6h)
WILLIAMS_R_PERIOD = 14
WILLIAMS_R_OVERBOUGHT = -20
WILLIAMS_R_OVERSOLD = -80

# Risk management
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_1d_13 = pd.Series(close_1d).ewm(span=ELDER_EMA_PERIOD, adjust=False, min_periods=ELDER_EMA_PERIOD).mean().values
    ema_1d_13_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_13)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator (6h) - Smoothed Moving Average (SMMA) approximation via EMA
    jaw = pd.Series(high).ewm(span=ALLIGATOR_JAW_PERIOD, adjust=False, min_periods=ALLIGATOR_JAW_PERIOD).mean().values
    jaw = pd.Series(jaw).ewm(span=8, adjust=False, min_periods=8).mean().values  # Additional 8-period smoothing
    
    teeth = pd.Series(low).ewm(span=ALLIGATOR_TEETH_PERIOD, adjust=False, min_periods=ALLIGATOR_TEETH_PERIOD).mean().values
    teeth = pd.Series(teeth).ewm(span=5, adjust=False, min_periods=5).mean().values  # Additional 5-period smoothing
    
    lips = pd.Series(close).ewm(span=ALLIGATOR_LIPS_PERIOD, adjust=False, min_periods=ALLIGATOR_LIPS_PERIOD).mean().values
    lips = pd.Series(lips).ewm(span=3, adjust=False, min_periods=3).mean().values  # Additional 3-period smoothing
    
    # Elder Ray (1d)
    bull_power = high - ema_1d_13_aligned  # High - EMA13
    bear_power = ema_1d_13_aligned - low   # EMA13 - Low
    
    # Williams %R (6h)
    highest_high = pd.Series(high).rolling(window=WILLIAMS_R_PERIOD, min_periods=WILLIAMS_R_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=WILLIAMS_R_PERIOD, min_periods=WILLIAMS_R_PERIOD).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
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
    
    # Warmup: max of all indicator lookbacks
    start = max(
        ALLIGATOR_JAW_PERIOD + 8,  # Jaw smoothing
        ALLIGATOR_TEETH_PERIOD + 5,  # Teeth smoothing
        ALLIGATOR_LIPS_PERIOD + 3,   # Lips smoothing
        WILLIAMS_R_PERIOD,
        ATR_PERIOD
    ) + 5
    
    for i in range(start, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_1d_13_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1 and close[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray confirmation from 1d
        strong_bull_power = bull_power[i] > 0  # Bulls in control
        strong_bear_power = bear_power[i] > 0  # Bears in control
        
        # Williams %R for entry timing
        oversold = williams_r[i] < WILLIAMS_R_OVERSOLD
        overbought = williams_r[i] > WILLIAMS_R_OVERBOUGHT
        
        # Entry conditions
        long_entry = bullish_alignment and strong_bull_power and oversold
        short_entry = bearish_alignment and strong_bear_power and overbought
        
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