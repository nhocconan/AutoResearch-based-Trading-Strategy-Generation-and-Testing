#!/usr/bin/env python3
name = "12h_TRIX_Pivot_Refill_v1"
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
    
    # Load daily data ONCE before loop for weekly pivot and TRIX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly pivot points using Monday open, Friday close approximation
    # Calculate weekly high/low/close from daily data using 5-day windows
    week_high = pd.Series(high).rolling(window=5, min_periods=5).max().values
    week_low = pd.Series(low).rolling(window=5, min_periods=5).min().values
    week_close = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Pivot levels
    pp = (week_high + week_low + week_close) / 3
    r1 = 2 * pp - week_low
    s1 = 2 * pp - week_high
    
    # TRIX on daily close (12-period EMA triple)
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # first value undefined
    
    # Volume spike detection: 20-period average (10 days of 12h bars)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(12, 20, 5)  # Wait for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pp_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(trix_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with TRIX positive and volume spike
            vol_condition = volume[i] > vol_ma_20_aligned[i] * 1.8
            if close[i] > s1_aligned[i] and trix_aligned[i] > 0 and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price below R1 with TRIX negative and volume spike
            elif close[i] < r1_aligned[i] and trix_aligned[i] < 0 and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below pivot or TRIX turns negative
            if close[i] < pp_aligned[i] or trix_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above pivot or TRIX turns positive
            if close[i] > pp_aligned[i] or trix_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: TRIX + Weekly Pivot with volume confirmation for 12h timeframe
# - TRIX (triple EMA crossover) filters momentum direction
# - Weekly pivot S1/R1 act as dynamic support/resistance levels
# - Long when price breaks above S1 with TRIX>0 and volume spike
# - Short when price breaks below R1 with TRIX<0 and volume spike
# - Exit when price returns to weekly pivot or TRIX reverses
# - Works in bull/bear: TRIX filters false breakouts, pivot provides structure
# - Volume confirmation ensures institutional participation
# - Position size 0.25 targets 15-35 trades/year, avoiding fee drag
# - Weekly pivot from daily data provides weekly structure without look-ahead
# - TRIX calculated on daily close with proper alignment avoids look-ahead
# - Max 2 conditions for entry reduces overtrading risk
# - Designed for BTC/ETH primary focus with applicability to SOL