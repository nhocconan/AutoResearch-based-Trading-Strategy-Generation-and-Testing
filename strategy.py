#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA34 trend filter and Camarilla pivot calculation.
- Entry: Long when price breaks above R1 with volume > 1.5x average AND price > 1d EMA34.
         Short when price breaks below S1 with volume > 1.5x average AND price < 1d EMA34.
- Exit: Opposite breakout (price < R1 for long, price > S1 for short) OR trend reversal.
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla levels provide high-probability intraday support/resistance.
- Volume confirmation ensures breakout legitimacy.
- 1d EMA34 filter aligns with higher timeframe trend to reduce counter-trend trades.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    pivot = (high + low + close) / 3
    range_val = high - low
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    return r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d Camarilla levels (R1, S1)
    camarilla_r1_1d = np.full(len(df_1d), np.nan)
    camarilla_s1_1d = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        r1, s1 = calculate_camarilla(df_1d['high'].iloc[i], df_1d['low'].iloc[i], df_1d['close'].iloc[i])
        camarilla_r1_1d[i] = r1
        camarilla_s1_1d[i] = s1
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 40  # Need sufficient data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_1d_aligned[i]) or 
            np.isnan(camarilla_s1_1d_aligned[i]) or
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        vol_ma = volume_ma[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = curr_volume > 1.5 * vol_ma
        
        # Exit conditions
        if position != 0:
            # Exit long: price falls below R1 OR trend reverses
            if position == 1:
                if curr_close < camarilla_r1_1d_aligned[i] or curr_close < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price rises above S1 OR trend reverses
            elif position == -1:
                if curr_close > camarilla_s1_1d_aligned[i] or curr_close > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions
        if position == 0:
            # Long: price breaks above R1 with volume confirmation AND bullish trend
            if curr_close > camarilla_r1_1d_aligned[i] and volume_confirmed and curr_close > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation AND bearish trend
            elif curr_close < camarilla_s1_1d_aligned[i] and volume_confirmed and curr_close < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0