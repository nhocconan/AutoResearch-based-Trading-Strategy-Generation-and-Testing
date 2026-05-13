#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator + 1d EMA34 trend filter + volume confirmation + ATR stoploss.
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > lips AND 1d EMA34 rising AND volume > 1.5x average.
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < lips AND 1d EMA34 falling AND volume > 1.5x average.
# Uses ATR(14) trailing stop (2.5x) for risk control. Discrete sizing 0.25.
# Uses 1d HTF for EMA34 trend filter and 1w HTF for regime (optional, not used here to keep simple).
# Target: 50-150 total trades over 4 years (12-37/year) on 12h.

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike_ATRStop_v1"
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
    
    # Get 12h data for Williams Alligator (Jaws=13, Teeth=8, Lips=5 SMAs with offsets)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate Alligator components (all based on 12h close)
    # Jaws: 13-period SMA, offset by 8 bars
    jaws_12h = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().values
    jaws_12h = np.roll(jaws_12h, 8)  # offset into future
    # Teeth: 8-period SMA, offset by 5 bars
    teeth_12h = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().values
    teeth_12h = np.roll(teeth_12h, 5)  # offset into future
    # Lips: 5-period SMA, offset by 3 bars
    lips_12h = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().values
    lips_12h = np.roll(lips_12h, 3)  # offset into future
    
    # Align 12h Alligator to 12h timeframe (wait for 12h bar to close)
    jaws_12h_aligned = align_htf_to_ltf(prices, df_12h, jaws_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe (wait for 1d bar to close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(jaws_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Alligator bullish (jaws < teeth < lips) AND price > lips AND 1d EMA34 rising AND volume > 1.5x average
            if (jaws_12h_aligned[i] < teeth_12h_aligned[i] < lips_12h_aligned[i] and
                close[i] > lips_12h_aligned[i] and
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Alligator bearish (jaws > teeth > lips) AND price < lips AND 1d EMA34 falling AND volume > 1.5x average
            elif (jaws_12h_aligned[i] > teeth_12h_aligned[i] > lips_12h_aligned[i] and
                  close[i] < lips_12h_aligned[i] and
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and
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