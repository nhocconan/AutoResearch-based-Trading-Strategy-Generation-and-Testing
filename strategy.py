#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout + 1w EMA50 trend filter + volume spike confirmation.
# Long when price breaks above 20-bar high AND close > 1w EMA50 AND volume > 2.0x 20-period average
# Short when price breaks below 20-bar low AND close < 1w EMA50 AND volume > 2.0x 20-period average
# Exit on opposite Donchian breakout or ATR trailing stop (2.5x)
# Uses 12h timeframe with 1w trend filter to avoid counter-trend trades, targeting 50-150 total trades over 4 years.
# Donchian channels provide objective breakout levels, 1w EMA50 filters intermediate trend, volume confirms breakout strength.

name = "12h_Donchian20_1wEMA50_VolumeSpike_v1"
timeframe = "12h"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian(20) channels on 12h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1w data for EMA50 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w close
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF arrays to 12h timeframe (wait for completed 1w bar)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current 12h volume > 2.0x 20-period average (strong spike confirmation)
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(50, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        # Bullish breakout: price > 20-bar high
        bullish_breakout = close[i] > highest_20[i]
        # Bearish breakout: price < 20-bar low
        bearish_breakout = close[i] < lowest_20[i]
        
        if position == 0:
            # LONG: bullish breakout AND close > 1w EMA50 AND volume spike
            if bullish_breakout and close[i] > ema50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: bearish breakout AND close < 1w EMA50 AND volume spike
            elif bearish_breakout and close[i] < ema50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
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
            # EXIT LONG: bearish breakout (price < 20-bar low) OR trailing stop hit
            breakout_exit = close[i] < lowest_20[i]
            trailing_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
            if breakout_exit or trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: bullish breakout (price > 20-bar high) OR trailing stop hit
            breakout_exit = close[i] > highest_20[i]
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
            if breakout_exit or trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals