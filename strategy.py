#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R1/S1 breakout with 1d volume spike and choppiness regime filter.
# Long when price breaks above R1 (bullish breakout) AND 1d volume > 1.5x 20-period average AND chop > 61.8 (ranging market -> mean reversion setup).
# Short when price breaks below S1 (bearish breakout) AND same volume/chop conditions.
# Uses ATR-based trailing stop (2.0x) for risk control.
# Designed to capture mean-reversion bounces in ranging markets with volume confirmation.
# Target: 20-40 trades/year.

name = "4h_Camarilla_R1S1_Breakout_1dVolume_Chop_v1"
timeframe = "4h"
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
    
    # Get 1d data for Camarilla pivot levels, volume, and choppiness
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate previous day's Camarilla levels (using prior day's OHLC)
    # Shift by 1 to use completed day's data
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan  # First day has no prior day
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Camarilla R1 and S1 levels
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    rangeprev = prev_high_1d - prev_low_1d
    camarilla_r1 = prev_close_1d + rangeprev * 1.1 / 12
    camarilla_s1 = prev_close_1d - rangeprev * 1.1 / 12
    
    # Calculate 1d volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    
    # Calculate Choppiness Index (CHOP) - range detector
    # CHOP = 100 * log10(sum(ATR1) / (n * (max(high) - min(low)))) / log10(n)
    # We'll use a simplified version: high-low range relative to ATR sum
    atr_1d = pd.Series(np.maximum(high_1d - low_1d, 
                                  np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                           np.abs(low_1d - np.roll(close_1d, 1))))).rolling(window=14, min_periods=14).mean().values
    atr_1d[0] = high_1d[0] - low_1d[0]  # First value
    
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Avoid division by zero
    chop = np.where(range_14 > 0, 
                    100 * np.log10(sum_atr_14 / (14 * range_14)) / np.log10(14), 
                    50)  # Neutral when range is zero
    
    # Align HTF arrays to 4h timeframe (wait for completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            # Carry forward tracking values when flat
            if i > 0 and position == 0:
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i-1]
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 AND volume spike AND chop > 61.8 (ranging market)
            if close[i] > camarilla_r1_aligned[i] and volume_spike_aligned[i] > 0.5 and chop_aligned[i] > 61.8:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price breaks below S1 AND volume spike AND chop > 61.8 (ranging market)
            elif close[i] < camarilla_s1_aligned[i] and volume_spike_aligned[i] > 0.5 and chop_aligned[i] > 61.8:
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