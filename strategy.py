#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
# Long when Alligator jaws (13-period SMMA) turns up, price > jaws, 1d EMA50 rising, and volume > 1.5x 20-period average.
# Short when jaws turn down, price < jaws, 1d EMA50 falling, and volume confirmation.
# Uses ATR(14) trailing stop (2.0x) for risk control.
# Williams Alligator identifies trend initiation and continuation via smoothed moving averages.
# 1d EMA50 ensures alignment with higher-timeframe trend, reducing whipsaws in choppy markets.
# Volume confirmation adds validity to trend-following signals.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h.

name = "6h_WilliamsAlligator_1dEMA50_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, period):
    """Smoothed Moving Average (SMMA) as used in Williams Alligator."""
    if len(source) < period:
        return np.full_like(source, np.nan, dtype=np.float64)
    sma = pd.Series(source).rolling(window=period, min_periods=period).mean().values
    smma_val = np.full_like(source, np.nan, dtype=np.float64)
    smma_val[period-1] = sma[period-1]
    for i in range(period, len(source)):
        smma_val[i] = (smma_val[i-1] * (period-1) + source[i]) / period
    return smma_val

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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe (wait for 1d bar to close)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: three SMMA lines
    # Jaws: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    jaws = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift the lines as per Alligator definition
    jaws_shifted = np.roll(jaws, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # First 8 values of jaws_shifted are invalid (rolled from end)
    jaws_shifted[:8] = np.nan
    # First 5 values of teeth_shifted are invalid
    teeth_shifted[:5] = np.nan
    # First 3 values of lips_shifted are invalid
    lips_shifted[:3] = np.nan
    
    # Calculate volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(jaws_shifted[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Jaws turning up (current > previous), price > jaws, 1d EMA50 rising, volume confirmation
            if (jaws_shifted[i] > jaws_shifted[i-1] and 
                close[i] > jaws_shifted[i] and 
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Jaws turning down (current < previous), price < jaws, 1d EMA50 falling, volume confirmation
            elif (jaws_shifted[i] < jaws_shifted[i-1] and 
                  close[i] < jaws_shifted[i] and 
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and 
                  volume_confirm[i]):
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