#!/usr/bin/env python3
"""
Experiment #8159: 6-hour Williams Alligator with 12-hour trend filter and volume confirmation.
Hypothesis: The Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and momentum,
while the 12-hour EMA filter ensures alignment with higher timeframe trend to avoid counter-trend trades.
Volume confirmation filters weak breakouts. This combination works in both bull and bear markets
by only taking trades in the direction of the 12-hour trend, reducing whipsaw during reversals.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8159_6w_alligator12h_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13    # Jaw (blue line)
ALLIGATOR_TEETH_PERIOD = 8   # Teeth (red line)
ALLIGATOR_LIPS_PERIOD = 5    # Lips (green line)
ALLIGATOR_JAW_SHIFT = 8
ALLIGATOR_TEETH_SHIFT = 5
ALLIGATOR_LIPS_SHIFT = 3
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
EMA_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_alligator_lines(data, jaw_period, teeth_period, lips_period, jaw_shift, teeth_shift, lips_shift):
    """Calculate Williams Alligator lines (SMMA with shifts)"""
    # Smoothed Moving Average (SMMA) approximation using EMA
    jaw = pd.Series(data).ewm(span=jaw_period, adjust=False).mean().shift(jaw_shift)
    teeth = pd.Series(data).ewm(span=teeth_period, adjust=False).mean().shift(teeth_shift)
    lips = pd.Series(data).ewm(span=lips_period, adjust=False).mean().shift(lips_shift)
    return jaw.values, teeth.values, lips.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    price_above_ema = close_12h > ema_12h  # True for bullish trend
    price_above_ema_aligned = align_htf_to_ltf(prices, df_12h, price_above_ema)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator lines
    jaw, teeth, lips = calculate_alligator_lines(
        close, 
        ALLIGATOR_JAW_PERIOD, 
        ALLIGATOR_TEETH_PERIOD, 
        ALLIGATOR_LIPS_PERIOD,
        ALLIGATOR_JAW_SHIFT,
        ALLIGATOR_TEETH_SHIFT,
        ALLIGATOR_LIPS_SHIFT
    )
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
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
    start = max(
        ALLIGATOR_JAW_PERIOD + ALLIGATOR_JAW_SHIFT,
        ALLIGATOR_TEETH_PERIOD + ALLIGATOR_TEETH_SHIFT,
        ALLIGATOR_LIPS_PERIOD + ALLIGATOR_LIPS_SHIFT,
        VOLUME_MA_PERIOD,
        ATR_PERIOD,
        EMA_PERIOD
    ) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_above_ema_aligned[i]):
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
        bull_bias = price_above_ema_aligned[i]   # 12h close above EMA50
        bear_bias = ~price_above_ema_aligned[i]  # 12h close below EMA50
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Alligator conditions: Lips > Teeth > Jaw = bullish alignment
        # Lips < Teeth < Jaw = bearish alignment
        bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        # Entry conditions
        long_entry = bull_bias and bullish_alignment and volume_confirmed
        short_entry = bear_bias and bearish_alignment and volume_confirmed
        
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