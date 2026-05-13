#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA50 trend filter and volume confirmation.
# Long when Alligator Lips cross above Teeth (bullish alignment) AND 1d EMA50 is rising AND volume > 1.5x 20-period average.
# Short when Alligator Lips cross below Teeth (bearish alignment) AND 1d EMA50 is falling AND volume > 1.5x 20-period average.
# Uses ATR(14) trailing stop (2.5x) for risk control.
# Williams Alligator identifies trend phases via smoothed medians; avoids whipsaws in choppy markets.
# 1d EMA50 ensures we trade with the dominant daily trend, reducing counter-trend entries.
# Volume confirmation adds validity to Alligator signals. Target: 50-150 total trades over 4 years (12-37/year) on 12h.

name = "12h_WilliamsAlligator_1dEMA50_Trend_Volume_v1"
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
    
    # Calculate Williams Alligator on 12h data (Jaw=13, Teeth=8, Lips=5)
    df_12h = get_htf_data(prices, '12h')
    median_12h = (df_12h['high'].values + df_12h['low'].values) / 2
    
    # Jaw: 13-period SMMA (smoothed moving average) of median, shifted 8 bars
    jaw_12h = pd.Series(median_12h).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    jaw_12h = np.roll(jaw_12h, 8)
    jaw_12h[:8] = np.nan
    
    # Teeth: 8-period SMMA of median, shifted 5 bars
    teeth_12h = pd.Series(median_12h).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    teeth_12h = np.roll(teeth_12h, 5)
    teeth_12h[:5] = np.nan
    
    # Lips: 5-period SMMA of median, shifted 3 bars
    lips_12h = pd.Series(median_12h).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    lips_12h = np.roll(lips_12h, 3)
    lips_12h[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe (wait for 12h bar to close)
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe (wait for 1d bar to close)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth (bullish alignment) AND 1d EMA50 rising AND volume confirmation
            if lips_12h_aligned[i] > teeth_12h_aligned[i] and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Lips < Teeth (bearish alignment) AND 1d EMA50 falling AND volume confirmation
            elif lips_12h_aligned[i] < teeth_12h_aligned[i] and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and volume_confirm[i]:
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