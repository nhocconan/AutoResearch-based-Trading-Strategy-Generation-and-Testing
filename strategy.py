#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
# Long when price breaks above 20-period Donchian high AND weekly pivot is bullish AND volume > 1.8x 20-period average.
# Short when price breaks below 20-period Donchian low AND weekly pivot is bearish AND volume > 1.8x 20-period average.
# Exit on opposite Donchian breakout or ATR(14) trailing stop (2.0x).
# Weekly pivot direction provides structural bias (bullish/bearish) from higher timeframe, reducing false breakouts in chop.
# Volume confirmation ensures breakout authenticity. Designed for 6h timeframe targeting 50-150 trades over 4 years.
# Works in bull markets via breakout continuation and in bear markets via fade at weekly resistance/support.

name = "6h_Donchian20_WeeklyPivot_VolumeSpike_v1"
timeframe = "6h"
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
    
    # Calculate Donchian channels (20-period) using previous bar to avoid look-ahead
    # Upper = max(high[-20:-1]), Lower = min(low[-20:-1])
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = np.roll(high_roll, 1)  # Shift to use previous bar's max
    donchian_low = np.roll(low_roll, 1)    # Shift to use previous bar's min
    donchian_high[0] = high[0]  # First bar: use current high
    donchian_low[0] = low[0]
    
    # Get weekly data for pivot direction (MTF)
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3, Bullish if close > P, Bearish if close < P
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_bullish = weekly_close > weekly_pivot  # True for bullish bias
    weekly_bearish = weekly_close < weekly_pivot   # True for bearish bias
    
    # Align HTF arrays to 6h timeframe (wait for completed weekly bar)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Volume filter: current 6h volume > 1.8x 20-period average (spike confirmation)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(50, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian high AND weekly bullish AND volume spike
            if close[i] > donchian_high[i] and weekly_bullish_aligned[i] > 0.5 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: price breaks below Donchian low AND weekly bearish AND volume spike
            elif close[i] < donchian_low[i] and weekly_bearish_aligned[i] > 0.5 and volume_filter[i]:
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
            # EXIT LONG: price breaks below Donchian low (opposite breakout) OR trailing stop hit
            breakout_exit = close[i] < donchian_low[i]
            trailing_stop = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
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
            # EXIT SHORT: price breaks above Donchian high (opposite breakout) OR trailing stop hit
            breakout_exit = close[i] > donchian_high[i]
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
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