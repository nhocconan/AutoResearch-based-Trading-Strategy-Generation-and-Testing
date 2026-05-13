#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation.
# Long when price breaks above R3 AND close > 1d EMA50 AND volume > 2.0x 20-period average.
# Short when price breaks below S3 AND close < 1d EMA50 AND volume > 2.0x 20-period average.
# Exit on opposite breakout or ATR(14) trailing stop (2.5x).
# Uses 12h timeframe with 1d trend filter for noise reduction, targeting 50-150 trades over 4 years.
# Camarilla R3/S3 levels provide stronger support/resistance than R1/S1, reducing false breakouts.
# EMA50 on 1d filters intermediate trend, volume confirms breakout authenticity.

name = "12h_Camarilla_R3_S3_Breakout_1dEMA50_VolumeSpike_v1"
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
    
    # Calculate Camarilla pivot levels for 12h: based on previous bar's OHLC
    # R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
    # Using previous bar to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # First bar: use current close
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    camarilla_range = prev_high - prev_low
    R3 = prev_close + 1.1 * camarilla_range / 4
    S3 = prev_close - 1.1 * camarilla_range / 4
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d close
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF arrays to 12h timeframe (wait for completed 1d bar)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current 12h volume > 2.0x 20-period average (spike confirmation)
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(50, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3 AND close > 1d EMA50 AND volume spike
            if close[i] > R3[i] and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: price breaks below S3 AND close < 1d EMA50 AND volume spike
            elif close[i] < S3[i] and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
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
            # EXIT LONG: price breaks below S3 (opposite breakout) OR trailing stop hit
            breakout_exit = close[i] < S3[i]
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
            # EXIT SHORT: price breaks above R3 (opposite breakout) OR trailing stop hit
            breakout_exit = close[i] > R3[i]
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