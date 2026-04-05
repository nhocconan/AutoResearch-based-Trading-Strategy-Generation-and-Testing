#!/usr/bin/env python3
"""
Experiment #8911: 6s Williams Alligator + Elder Ray + 1d Trend Filter (Elder Ray + Alligator)
Hypothesis: Elder Ray measures bull/bear power via EMA13; Williams Alligator (jaw/teeth/lips) filters false signals.
Combines momentum (Alligator) with strength (Elder Ray) for robust entries. Works in bull via Elder Ray >0 + Alligator bullish alignment,
and in bear via Elder Ray <0 + Alligator bearish alignment. Uses 1d EMA200 for trend filter to avoid counter-trend trades.
Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag. Uses 6h timeframe as primary.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_8911_6h_elder_ray_alligator_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ELDER_RAY_EMA = 13
ALLIGATOR_JAW = 13
ALLIGATOR_TEETH = 8
ALLIGATOR_LIPS = 5
TREND_FILTER_EMA = 200
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ema(arr, period):
    """Calculate EMA using Wilder's smoothing (alpha = 1/period)"""
    return pd.Series(arr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def calculate_wma(arr, period):
    """Calculate Weighted Moving Average"""
    df = pd.Series(arr)
    weights = np.arange(1, period + 1)
    return df.rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    ).values

def calculate_alligator(high, low, close):
    """Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs of median price"""
    median_price = (high + low) / 2
    jaw = calculate_wma(median_price, ALLIGATOR_JAW)
    teeth = calculate_wma(median_price, ALLIGATOR_TEETH)
    lips = calculate_wma(median_price, ALLIGATOR_LIPS)
    return jaw, teeth, lips

def calculate_elder_ray(close, ema_period):
    """Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA"""
    ema = calculate_ema(close, ema_period)
    bull_power = high - ema  # Will be set after high is defined
    bear_power = low - ema   # Will be set after low is defined
    return ema, bull_power, bear_power

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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_1d_200 = calculate_ema(close_1d, TREND_FILTER_EMA)
    
    # Price relative to 1d EMA200: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_1d > ema_1d_200, 1, 
                     np.where(close_1d < ema_1d_200, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Extract price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Elder Ray components (using EMA13)
    ema_13 = calculate_ema(close, ELDER_RAY_EMA)
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate Alligator components
    jaw, teeth, lips = calculate_alligator(high, low, close)
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ELDER_RAY_EMA, ALLIGATOR_JAW, TREND_FILTER_EMA, ATR_PERIOD) + 1
    
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
        
        # Determine market bias from 1d EMA200
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d price above EMA200
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d price below EMA200
        
        # Elder Ray: bull power > 0 and bear power < 0 indicates strength
        elder_bull = bull_power[i] > 0
        elder_bear = bear_power[i] < 0
        
        # Williams Alligator: 
        # Bullish alignment: Lips > Teeth > Jaw (all above)
        # Bearish alignment: Lips < Teeth < Jaw (all below)
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Entry conditions
        long_entry = bull_bias and elder_bull and alligator_bullish
        short_entry = bear_bias and elder_bear and alligator_bearish
        
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
</s>