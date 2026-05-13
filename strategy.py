#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
# Long when Alligator jaws (13-period smoothed median) crosses above teeth (8-period smoothed median)
# AND price > lips (5-period smoothed median) AND 1d EMA50 rising AND volume > 1.5x average.
# Short when jaws cross below teeth AND price < lips AND 1d EMA50 falling AND volume > 1.5x average.
# Uses ATR(14) trailing stop (2.5x) for risk control. Discrete sizing 0.25.
# Alligator identifies trend initiation/continuation, 1d EMA50 filters primary trend, volume confirms strength.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h.

name = "12h_WilliamsAlligator_1dEMA50_Volume_ATRStop_v1"
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
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Williams Alligator (based on median price)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    median_1d = (high_1d + low_1d) / 2.0
    
    # Williams Alligator lines: jaws (13), teeth (8), lips (5) - all smoothed with 3-period offset
    def smoothed_mma(series, period):
        # Smoothed Moving Average (SMMA) - equivalent to RMA/Wilder's
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
        # Convert SMA to SMMA: first value = SMA, then recursive: SMMA = (prev*(period-1) + current) / period
        smma = np.full_like(series, np.nan, dtype=float)
        smma[period-1] = sma[period-1]  # First valid value
        for i in range(period, len(series)):
            if not np.isnan(sma[i]) and not np.isnan(smma[i-1]):
                smma[i] = (smma[i-1] * (period-1) + sma[i]) / period
        return smma
    
    jaws = smoothed_mma(median_1d, 13)  # Jaw line (13-period)
    teeth = smoothed_mma(median_1d, 8)   # Teeth line (8-period)
    lips = smoothed_mma(median_1d, 5)    # Lips line (5-period)
    
    # Align 1d Alligator lines to 12h timeframe (wait for 1d bar to close)
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 1d data for EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe (wait for 1d bar to close)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Jaws cross above teeth AND price > lips AND 1d EMA50 rising AND volume > 1.5x average
            if (jaws_aligned[i] > teeth_aligned[i] and jaws_aligned[i-1] <= teeth_aligned[i-1] and
                close[i] > lips_aligned[i] and 
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Jaws cross below teeth AND price < lips AND 1d EMA50 falling AND volume > 1.5x average
            elif (jaws_aligned[i] < teeth_aligned[i] and jaws_aligned[i-1] >= teeth_aligned[i-1] and
                  close[i] < lips_aligned[i] and 
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and 
                  volume[i] > 1.5 * avg_volume[i]):
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
            # EXIT LONG: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
            if trailing_stop:
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
            # EXIT SHORT: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
            if trailing_stop:
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