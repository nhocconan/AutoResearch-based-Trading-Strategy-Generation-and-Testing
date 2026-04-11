#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v20
# Strategy: 4-hour Camarilla pivot breakout with 1-day volume confirmation and choppiness regime filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Price breaking above/below Camarilla pivot levels (H3/L3) from the prior day,
# confirmed by volume spike (>2x average) and filtered by choppiness index (<61.8 for trending markets),
# captures institutional breakouts in both bull and bear markets. The regime filter avoids whipsaws
# in sideways conditions, improving robustness.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v20"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC for Camarilla calculation
    # Use prior day's data to avoid look-ahead (current day still forming)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use prior completed day's data
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # Set first day's prior values to NaN (no prior day)
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Camarilla levels: H3, L3 (most relevant for breakouts)
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    camarilla_h3 = close_1d_prev + 1.1 * (high_1d_prev - low_1d_prev) / 4
    camarilla_l3 = close_1d_prev - 1.1 * (high_1d_prev - low_1d_prev) / 4
    
    # Align to 4h timeframe (wait for prior day to complete)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 4h volume confirmation: current volume > 2x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_avg_20)
    
    # Choppiness index regime filter (using daily data)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    # We want trending markets: CHOP < 61.8
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True range
    tr1 = high_1d_series - low_1d_series
    tr2 = abs(high_1d_series - close_1d_series.shift(1))
    tr3 = abs(low_1d_series - close_1d_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR(14)
    atr_14 = tr.rolling(window=14, min_periods=14).sum() / 14
    
    # Chop calculation: 100 * log10(sum(TR14)/ (ATR14 * 14)) / log10(14)
    sum_tr14 = tr.rolling(window=14, min_periods=14).sum()
    chop = 100 * (np.log10(sum_tr14 / (atr_14 * 14)) / np.log10(14))
    chop_values = chop.values
    
    # Align chop to 4h: we want chop < 61.8 (trending condition)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    chop_filter = chop_aligned < 61.8  # trending market
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_confirm[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        bull_breakout = close[i] > camarilla_h3_aligned[i]
        bear_breakout = close[i] < camarilla_l3_aligned[i]
        
        # Entry logic: breakout + volume + trend regime
        if bull_breakout and vol_confirm[i] and chop_filter[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif bear_breakout and vol_confirm[i] and chop_filter[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout with volume confirmation
        elif position == 1 and bear_breakout and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bull_breakout and vol_confirm[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals