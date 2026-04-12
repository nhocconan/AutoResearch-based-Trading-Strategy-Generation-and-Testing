#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_12h_camarilla_breakout_with_volume_and_chop
# Uses daily Camarilla pivot levels (from 12h data) on 4h chart.
# Long when price breaks above R4 with volume confirmation (volume > 2.0x 20-period avg) and chop > 61.8 (range).
# Short when price breaks below S4 with volume confirmation and chop > 61.8.
# Exits when price returns to the 12h close (mean reversion).
# Designed for low trade frequency (target: 15-35 trades/year) to minimize fee drag.
# Works in ranging markets via mean reversion and avoids false breakouts in trends.

name = "4h_12h_camarilla_breakout_with_volume_and_chop"
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
    
    # Get 12h data for Chop calculation and Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Chop index (12h)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original length
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log10(sum(TR,14) / (max(HH,14) - min(LL,14))) / log10(14)
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    denominator = highest_high - lowest_low
    chop = np.where(denominator > 0, 100 * np.log10(sum_tr / denominator) / np.log10(14), 50)
    
    # Align Chop to 4h
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    # We need daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    # R4 = Close + 1.5 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    
    # First value will be invalid due to roll, set to NaN
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_R4 = close_1d_prev + 1.5 * (high_1d_prev - low_1d_prev)
    camarilla_S4 = close_1d_prev - 1.5 * (high_1d_prev - low_1d_prev)
    # Exit level: previous day's close
    camarilla_exit = close_1d_prev
    
    # Align daily Camarilla levels to 4h timeframe
    camarilla_R4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R4)
    camarilla_S4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S4)
    camarilla_exit_aligned = align_htf_to_ltf(prices, df_1d, camarilla_exit)
    
    # Volume confirmation: volume > 2.0 * 20-period average (4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(camarilla_R4_aligned[i]) or np.isnan(camarilla_S4_aligned[i]) or 
            np.isnan(camarilla_exit_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Range condition: Chop > 61.8 (ranging market)
        if chop_aligned[i] <= 61.8:
            # Hold current position if not in range
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above R4
        if close[i] > camarilla_R4_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below S4
        elif close[i] < camarilla_S4_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to previous day's close (mean reversion)
        elif position == 1 and close[i] <= camarilla_exit_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= camarilla_exit_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals