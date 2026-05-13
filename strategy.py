#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator + 1d EMA50 trend filter + volume confirmation (>1.3x avg volume). 
# The Alligator (jaw/teeth/lips) identifies trend absence/presence via SMAs with offsets.
# Long when lips > teeth > jaw (bullish alignment) + price > lips + 1d EMA50 up + volume spike.
# Short when lips < teeth < jaw (bearish alignment) + price < lips + 1d EMA50 down + volume spike.
# Uses ATR(20) trailing stop (2.0x) for risk control. Discrete sizing 0.25.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
# Alligator filters choppy markets; EMA50 ensures higher timeframe trend alignment.
# Works in bull markets via trend-following entries and in bear markets via shorting with trend filter.

name = "6h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_ATRStop_v1"
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
    
    # Calculate ATR(20) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Williams Alligator: jaw(13,8), teeth(8,5), lips(5,3) - all SMAs with forward shift
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # shift forward by 8 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # shift forward by 5 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # shift forward by 3 bars
    
    jaw = jaw.values
    teeth = teeth.values
    lips = lips.values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe (wait for daily bar to close)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment) AND price > lips AND 1d EMA50 up AND volume > 1.3x average
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > lips[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and 
                volume[i] > 1.3 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Lips < Teeth < Jaw (bearish alignment) AND price < lips AND 1d EMA50 down AND volume > 1.3x average
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < lips[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and 
                  volume[i] > 1.3 * avg_volume[i]):
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
            # EXIT LONG: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
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
            # EXIT SHORT: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
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