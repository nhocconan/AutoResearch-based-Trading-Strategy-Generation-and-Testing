#!/usr/bin/env python3
# Hypothesis: 1h EMA crossover with 4h Supertrend regime filter and volume spike confirmation.
# Long when 1h EMA20 crosses above EMA50 AND 4h Supertrend is bullish AND volume > 1.5x average.
# Short when 1h EMA20 crosses below EMA50 AND 4h Supertrend is bearish AND volume > 1.5x average.
# Uses ATR(14) trailing stop (2.5x) for risk control. Discrete sizing 0.20.
# Target: 60-150 total trades over 4 years (15-37/year) on 1h.

name = "1h_EMA20_50_Cross_4hSupertrend_VolumeSpike_ATRStop_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1h EMA20 and EMA50 for crossover signals
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 4h data for Supertrend regime filter
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Supertrend on 4h
    # ATR period 10, multiplier 3.0
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_4h = pd.Series(tr_4h).rolling(window=10, min_periods=10).mean().values
    
    # Basic upper and lower bands
    hl2_4h = (high_4h + low_4h) / 2
    upper_band_4h = hl2_4h + (3.0 * atr_4h)
    lower_band_4h = hl2_4h - (3.0 * atr_4h)
    
    # Initialize Supertrend arrays
    supertrend_4h = np.full_like(close_4h, np.nan)
    direction_4h = np.full_like(close_4h, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Calculate Supertrend
    for i in range(len(close_4h)):
        if i == 0:
            supertrend_4h[i] = hl2_4h[i]
            direction_4h[i] = 1  # Start with uptrend
        else:
            if close_4h[i-1] > supertrend_4h[i-1]:
                # Previous close was above previous Supertrend
                upper_band_4h[i] = min(upper_band_4h[i], upper_band_4h[i-1])
                if close_4h[i] <= upper_band_4h[i]:
                    supertrend_4h[i] = upper_band_4h[i]
                    direction_4h[i] = -1  # Downtrend
                else:
                    supertrend_4h[i] = upper_band_4h[i]
                    direction_4h[i] = 1   # Uptrend
            else:
                # Previous close was below previous Supertrend
                lower_band_4h[i] = max(lower_band_4h[i], lower_band_4h[i-1])
                if close_4h[i] >= lower_band_4h[i]:
                    supertrend_4h[i] = lower_band_4h[i]
                    direction_4h[i] = 1   # Uptrend
                else:
                    supertrend_4h[i] = lower_band_4h[i]
                    direction_4h[i] = -1  # Downtrend
    
    # Align 4h Supertrend direction to 1h timeframe (wait for 4h bar to close)
    direction_4h_aligned = align_htf_to_ltf(prices, df_4h, direction_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(ema_20[i]) or np.isnan(ema_50[i]) or 
            np.isnan(direction_4h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Check for EMA crossover
            ema_cross_up = (ema_20[i] > ema_50[i]) and (ema_20[i-1] <= ema_50[i-1])
            ema_cross_down = (ema_20[i] < ema_50[i]) and (ema_20[i-1] >= ema_50[i-1])
            
            # LONG: EMA20 crosses above EMA50 AND 4h Supertrend bullish AND volume > 1.5x average
            if (ema_cross_up and 
                direction_4h_aligned[i] == 1 and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.20
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: EMA20 crosses below EMA50 AND 4h Supertrend bearish AND volume > 1.5x average
            elif (ema_cross_down and 
                  direction_4h_aligned[i] == -1 and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.20
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.20
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.20
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals