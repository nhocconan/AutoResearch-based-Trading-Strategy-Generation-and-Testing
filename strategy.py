#!/usr/bin/env python3
"""
Experiment #9399: 6h Williams Alligator + Elder Ray with 12h trend filter.
Hypothesis: Williams Alligator identifies trend presence (jaws/teeth/lips alignment),
Elder Ray measures bull/bear power strength, and 12h EMA filter ensures directional alignment.
Works in bull markets via strong bull power and aligned jaws, in bear via bear power and reversed alignment.
Targets 100-200 total trades over 4 years (25-50/year) to balance opportunity and cost.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_9399_6h_alligator_elder_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD_JAW = 13   # Smoothed SMA (13-period)
ALLIGATOR_PERIOD_TEETH = 8  # Smoothed SMA (8-period)
ALLIGATOR_PERIOD_LIPS = 5   # Smoothed SMA (5-period)
ELDER_RAY_PERIOD = 13       # EMA for Elder Ray
TREND_PERIOD = 50           # 12h EMA for trend filter
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def smma(series, period):
    """Smoothed Moving Average (SMMA) - Wilder's smoothing"""
    return pd.Series(series).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

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
    
    # Williams Alligator: three smoothed SMAs
    jaw = smma(high, ALLIGATOR_PERIOD_JAW)   # Typically uses median price, but high works for trend
    teeth = smma(high, ALLIGATOR_PERIOD_TEETH)
    lips = smma(high, ALLIGATOR_PERIOD_LIPS)
    
    # Elder Ray: Bull Power = High - EMA, Bear Power = EMA - Low
    ema_close = pd.Series(close).ewm(span=ELDER_RAY_PERIOD, adjust=False, min_periods=ELDER_RAY_PERIOD).mean().values
    bull_power = high - ema_close
    bear_power = ema_close - low
    
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
        # Mouth open: lips outside teeth/jaw indicating trend
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        # Alligator sleeping: all lines intertwined (no trend)
        jaw_teeth_close = np.abs(jaw[i] - teeth[i]) < (0.001 * close[i])
        teeth_lips_close = np.abs(teeth[i] - lips[i]) < (0.001 * close[i])
        alligator_sleeping = jaw_teeth_close and teeth_lips_close
        
        # Elder Ray conditions
        strong_bull_power = bull_power[i] > 0 and bull_power[i] > np.mean(bull_power[max(0, i-20):i+1])
        strong_bear_power = bear_power[i] > 0 and bear_power[i] > np.mean(bear_power[max(0, i-20):i+1])
        
        # Determine market bias from 12h EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 12h price above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 12h price below EMA50
        
        # Entry conditions
        # Long: Alligator trending up (lips > teeth > jaw) + bull power + bullish bias
        long_entry = (not alligator_sleeping and 
                     lips_above_teeth and teeth_above_jaw and 
                     strong_bull_power and bull_bias)
        
        # Short: Alligator trending down (lips < teeth < jaw) + bear power + bearish bias
        short_entry = (not alligator_sleeping and 
                      lips_below_teeth and teeth_below_jaw and 
                      strong_bear_power and bear_bias)
        
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