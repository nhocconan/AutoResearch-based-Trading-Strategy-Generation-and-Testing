#!/usr/bin/env python3
"""
Experiment #8339: 6-hour Williams Alligator with 12h/1d trend filter.
Hypothesis: In trending markets (60-80% of time), Williams Alligator (3 SMAs) provides clear trend direction.
We use 12h and 1d timeframes to filter for strong trends: price above/both 12h and 1d EMA50 for longs,
below both for shorts. This avoids whipsaw in sideways markets. Target: 50-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8339_6w_alligator_12h1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13   # SMMA(13, 8)
ALLIGATOR_TEETH_PERIOD = 8  # SMMA(8, 5)
ALLIGATOR_LIPS_PERIOD = 5   # SMMA(5, 3)
EMA_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SIGNAL_SIZE = 0.25

def smma(series, period):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    return pd.Series(series).ewm(alpha=1/period, adjust=False).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h and 1d EMA50 for trend filter
    close_12h = df_12h['close'].values
    close_1d = df_1d['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    
    # Trend: 1 = bullish (price > both EMAs), -1 = bearish (price < both EMAs), 0 = mixed/no trend
    bullish_trend = (close_12h > ema_12h) & (close_1d > ema_1d)
    bearish_trend = (close_12h < ema_12h) & (close_1d < ema_1d)
    trend_filter = np.where(bullish_trend, 1, np.where(bearish_trend, -1, 0))
    trend_filter_12h = align_htf_to_ltf(prices, df_12h, trend_filter)
    trend_filter_1d = align_htf_to_ltf(prices, df_1d, trend_filter)
    # Require both timeframes to agree
    trend_agreement = (trend_filter_12h == trend_filter_1d) & (trend_filter_12h != 0)
    trend_direction = trend_filter_12h  # same as trend_filter_1d when agreed
    
    # Calculate Williams Alligator on 6h
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # SMMA (Smoothed Moving Average) for Alligator lines
    jaw = smma(close, ALLIGATOR_JAW_PERIOD)    # Blue line: 13-period, 8 bars ahead
    teeth = smma(close, ALLIGATOR_TEETH_PERIOD) # Red line: 8-period, 5 bars ahead
    lips = smma(close, ALLIGATOR_LIPS_PERIOD)   # Green line: 5-period, 3 bars ahead
    
    # Alligator signals: 
    # - Mouth open (trending): jaws, teeth, lips are separated and ordered
    # - Mouth closed (sleeping): lines are intertwined
    # For trend following: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
    bullish_alligator = (lips > teeth) & (teeth > jaw)
    bearish_alligator = (lips < teeth) & (teeth < jaw)
    
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
    start = max(ALLIGATOR_JAW_PERIOD, ALLIGATOR_TEETH_PERIOD, ALLIGATOR_LIPS_PERIOD, EMA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if trend data not available
        if np.isnan(trend_direction[i]):
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
        
        # Determine Alligator signal
        bullish_alli = bullish_alligator[i] if not np.isnan(bullish_alligator[i]) else False
        bearish_alli = bearish_alligator[i] if not np.isnan(bearish_alligator[i]) else False
        
        # Determine trend direction from agreement of 12h and 1d
        bullish_trend = trend_direction[i] == 1
        bearish_trend = trend_direction[i] == -1
        
        # Entry conditions: Alligator alignment + trend filter agreement
        long_entry = bullish_alli and bullish_trend
        short_entry = bearish_alli and bearish_trend
        
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