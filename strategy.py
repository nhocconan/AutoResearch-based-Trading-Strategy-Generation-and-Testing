#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator + Elder Ray combination with 1d volume spike filter.
# Long when Alligator is bullish (jaw < teeth < lips) AND Bull Power > 0 AND volume > 2.0x 20-period average.
# Short when Alligator is bearish (jaw > teeth > lips) AND Bear Power < 0 AND volume > 2.0x 20-period average.
# Exit on ATR(21) trailing stop (2.5x). Uses 12h primary timeframe and 1d HTF for trend/power alignment.
# Williams Alligator identifies trend structure via smoothed medians, Elder Ray measures bull/bear power via EMA13.
# Volume spike confirms breakout authenticity. Designed for BTC/ETH with strict entry to avoid overtrading (target: 12-37 trades/year).

name = "12h_WilliamsAlligator_ElderRay_1dVolumeSpike_v1"
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
    
    # Calculate ATR(21) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=21, min_periods=21).mean().values
    
    # Get 1d data for EMA13 (Elder Ray) and volume (MTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA13 on 1d close for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power on 1d
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Calculate Williams Alligator on 1d (smoothed medians)
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars
    # Lips: 5-period SMMA shifted 3 bars
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) >= period:
            # First value is simple SMA
            result[period-1] = np.mean(arr[:period])
            # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_1d = smma(close_1d, 13)
    teeth_1d = smma(close_1d, 8)
    lips_1d = smma(close_1d, 5)
    
    # Shift the lines as per Alligator definition
    jaw_1d = np.roll(jaw_1d, 8)
    teeth_1d = np.roll(teeth_1d, 5)
    lips_1d = np.roll(lips_1d, 3)
    # Set NaN for rolled values
    jaw_1d[:8] = np.nan
    teeth_1d[:5] = np.nan
    lips_1d[:3] = np.nan
    
    # Align HTF arrays to 12h timeframe (wait for completed 1d bar)
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # Volume filter: current 12h volume > 2.0x 20-period average (spike confirmation)
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or np.isnan(lips_1d_aligned[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_12h[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Alligator bullish (jaw < teeth < lips) AND Bull Power > 0 AND volume spike
            if (jaw_1d_aligned[i] < teeth_1d_aligned[i] < lips_1d_aligned[i] and 
                bull_power_1d_aligned[i] > 0 and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Alligator bearish (jaw > teeth > lips) AND Bear Power < 0 AND volume spike
            elif (jaw_1d_aligned[i] > teeth_1d_aligned[i] > lips_1d_aligned[i] and 
                  bear_power_1d_aligned[i] < 0 and volume_filter[i]):
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