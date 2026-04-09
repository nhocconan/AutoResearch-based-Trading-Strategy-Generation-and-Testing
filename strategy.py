#!/usr/bin/env python3
# 4h_camarilla_pivot_volume_chop_v2
# Hypothesis: 4h strategy using 1d Camarilla pivot levels (L3/H3) with volume confirmation (>1.5x 20-period average) and 12h choppiness regime filter (CHOP > 61.8 = range). Enters long at L3 with volume confirmation in ranging markets; short at H3 with volume confirmation in ranging markets. Uses discrete position sizing (0.25) to limit fee drag. Designed for low turnover (target: 20-50 trades/year) to work in both bull and bear markets by mean-reverting at proven intraday support/resistance levels during range-bound conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_chop(high, low, close, period=14):
    """Calculate Choppiness Index"""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    atr = pd.Series(np.maximum(np.abs(high - low), np.maximum(np.abs(high - close[:-1]), np.abs(low - close[:-1]))).rolling(window=1, min_periods=1).sum()).rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(atr / (highest_high - lowest_low)) / np.log10(period)
    return chop.values

name = "4h_camarilla_pivot_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 12h HTF chop regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    chop_12h = calculate_chop(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # 1d HTF for Camarilla pivot levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # L3 = C - (H-L)*1.1/4, H3 = C + (H-L)*1.1/4
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    rang = prev_high - prev_low
    L3 = prev_close - (rang * 1.1 / 4)
    H3 = prev_close + (rang * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (using previous day's levels)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(chop_12h_aligned[i]) or
            np.isnan(L3_aligned[i]) or np.isnan(H3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        ranging_market = chop_12h_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price crosses above H3 (take profit) or below L3 (stop)
            if close[i] > H3_aligned[i] or close[i] < L3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses below L3 (take profit) or above H3 (stop)
            if close[i] < L3_aligned[i] or close[i] > H3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with volume confirmation and in ranging market
            if volume_confirmed and ranging_market:
                # Long at L3 support
                if close[i] <= L3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short at H3 resistance
                elif close[i] >= H3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals