#!/usr/bin/env python3
"""
6h Williams Alligator + Elder Ray
Hypothesis: Alligator identifies trend direction and strength (jaws-teeth-lips alignment).
Elder Ray measures bull/bear power via EMA(13). Long when bull power > 0 and price above teeth.
Short when bear power < 0 and price below teeth. Uses 12h trend filter for higher timeframe alignment.
Works in trends (trend follow) and ranges (fade at extremes). Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14339_6h_alligator_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 50-period EMA for 12h trend filter
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator (13,8,5 with 8,5,3 offsets)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_offset = 8
    teeth_offset = 5
    lips_offset = 3
    
    # Calculate SMAs with offsets
    sma_jaw = pd.Series(high + low + close).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    sma_teeth = pd.Series(high + low + close).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    sma_lips = pd.Series(high + low + close).rolling(window=lips_period, min_periods=lips_period).mean().values
    
    # Apply offsets (shift forward)
    jaw = np.roll(sma_jaw, -jaw_offset)
    teeth = np.roll(sma_teeth, -teeth_offset)
    lips = np.roll(sma_lips, -lips_offset)
    
    # For first offset periods, use unshifted values to avoid look-ahead
    jaw[:jaw_offset] = sma_jaw[:jaw_offset]
    teeth[:teeth_offset] = sma_teeth[:teeth_offset]
    lips[:lips_offset] = sma_lips[:lips_offset]
    
    # Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(jaw_period + jaw_offset, teeth_period + teeth_offset, lips_period + lips_offset, 13) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_12h_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price crosses below teeth OR stoploss
            if close[i] <= teeth[i] or close[i] <= entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above teeth OR stoploss
            if close[i] >= teeth[i] or close[i] >= entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Alligator alignment + Elder Ray + 12h trend filter
            # Alligator is aligned when jaws > teeth > lips (uptrend) or jaws < teeth < lips (downtrend)
            alligator_long = jaw[i] > teeth[i] and teeth[i] > lips[i]
            alligator_short = jaw[i] < teeth[i] and teeth[i] < lips[i]
            
            # Elder Ray: bull power > 0 and bear power < 0 for confirmation
            long_setup = alligator_long and (bull_power[i] > 0) and (close[i] > ema_12h_aligned[i])
            short_setup = alligator_short and (bear_power[i] < 0) and (close[i] < ema_12h_aligned[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals