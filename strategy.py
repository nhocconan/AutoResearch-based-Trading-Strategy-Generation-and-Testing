#!/usr/bin/env python3
"""
Experiment #12359: 6h Williams Alligator + Elder Ray + Volume Confirmation
Hypothesis: Williams Alligator (JAW/TEETH/LIPS) identifies trend direction and alignment.
Elder Ray (Bull/Bear Power) measures momentum behind the trend. Volume confirms strength.
Works in bull via bullish alignment + bullish power, in bear via bearish alignment + bearish power.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12359_6w_alligator_elder_ray_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_PERIOD = 13  # base period for SMAs
ELDER_RAY_POWER_PERIOD = 13
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_sma(arr, period):
    """Calculate Simple Moving Average"""
    return pd.Series(arr).rolling(window=period, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend context (optional filter)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma_1d_50 = calculate_sma(close_1d, 50)
    sma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_50)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator: SMAs with specific offsets
    jaw = calculate_sma(close, ALLIGATOR_PERIOD * 3)  # 13*3 = 39
    teeth = calculate_sma(close, ALLIGATOR_PERIOD * 2)  # 13*2 = 26
    lips = calculate_sma(close, ALLIGATOR_PERIOD)       # 13*1 = 13
    
    # Shift jaws/teeth/lips forward by their respective periods
    jaw = np.roll(jaw, ALLIGATOR_PERIOD * 2)   # shift by 26
    teeth = np.roll(teeth, ALLIGATOR_PERIOD)   # shift by 13
    # lips not shifted
    
    # Elder Ray Power
    bull_power = high - jaws  # Actually: High - EMA(13) but using JAW as proxy
    bear_power = lips - low   # Lips - Low
    # Better: use EMA(13) for Elder Ray
    ema_13 = calculate_sma(close, ELDER_RAY_POWER_PERIOD)  # SMA as proxy for EMA
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    volume_ma = calculate_sma(volume, VOLUME_MA_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Warmup: max of all periods plus shifts
    start = max(ALLIGATOR_PERIOD * 3, ALLIGATOR_PERIOD * 2, ALLIGATOR_PERIOD,
                ELDER_RAY_POWER_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 5
    
    for i in range(start, n):
        # Skip if 1d SMA not available (context filter)
        if not np.isnan(sma_1d_50_aligned[i]):
            # Optional: only trade in direction of 1d trend
            market_bullish = close[i] > sma_1d_50_aligned[i]
            market_bearish = close[i] < sma_1d_50_aligned[i]
        else:
            market_bullish = True  # allow both if no 1d data
            market_bearish = True
        
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
        
        # Williams Alligator alignment
        # Alligator is aligned when: Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        
        # Handle potential NaN from rolling
        if np.isnan(lips_val) or np.isnan(teeth_val) or np.isnan(jaw_val):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        bullish_alignment = (lips_val > teeth_val) and (teeth_val > jaw_val)
        bearish_alignment = (lips_val < teeth_val) and (teeth_val < jaw_val)
        
        # Elder Ray Power
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        
        strong_bull_power = bull_power_val > 0 and bull_power_val > np.nanmean(bull_power[max(0, i-50):i]) if i >= 50 else bull_power_val > 0
        strong_bear_power = bear_power_val > 0 and bear_power_val > np.nanmean(bear_power[max(0, i-50):i]) if i >= 50 else bear_power_val > 0
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bullish_alignment and strong_bull_power and volume_ok and market_bullish
        short_entry = bearish_alignment and strong_bear_power and volume_ok and market_bearish
        
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