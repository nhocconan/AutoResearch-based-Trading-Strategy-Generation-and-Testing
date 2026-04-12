#!/usr/bin/env python3
"""
4h_1D_Chaikin_Money_Flow_Strategy
Hypothesis: Chaikin Money Flow (CMF) on daily timeframe detects institutional accumulation/distribution.
Long when CMF > 0.15 with price above 4h EMA50, short when CMF < -0.15 with price below EMA50.
Uses volume-weighted money flow to capture smart money moves, effective in both bull (accumulation) and bear (distribution) markets.
Low trade frequency (~25/year) reduces fee drag while capturing significant moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1D_Chaikin_Money_Flow_Strategy"
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
    
    # === DAILY DATA FOR CHAIKIN MONEY FLOW ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Money Flow Multiplier: ((Close - Low) - (High - Close)) / (High - Low)
    # Avoid division by zero
    hl_range = high_1d - low_1d
    hl_range = np.where(hl_range == 0, 1, hl_range)  # Replace zeros with 1 to avoid div/0
    mf_multiplier = ((close_1d - low_1d) - (high_1d - close_1d)) / hl_range
    
    # Money Flow Volume
    mf_volume = mf_multiplier * volume_1d
    
    # Chaikin Money Flow (20-period)
    mf_volume_sum = pd.Series(mf_volume).rolling(window=20, min_periods=20).sum().values
    volume_sum = pd.Series(volume_1d).rolling(window=20, min_periods=20).sum().values
    cmf = np.where(volume_sum != 0, mf_volume_sum / volume_sum, 0)
    
    # Align CMF to 4h timeframe (wait for daily bar to close)
    cmf_aligned = align_htf_to_ltf(prices, df_1d, cmf)
    
    # === 4H INDICATORS ===
    # EMA50 for trend filter
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation (4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(cmf_aligned[i]) or np.isnan(ema50[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long setup: CMF positive (accumulation) + price above EMA50 + volume confirmation
        long_setup = (cmf_aligned[i] > 0.15) and (close[i] > ema50[i]) and (vol_ratio[i] > 1.2)
        
        # Short setup: CMF negative (distribution) + price below EMA50 + volume confirmation
        short_setup = (cmf_aligned[i] < -0.15) and (close[i] < ema50[i]) and (vol_ratio[i] > 1.2)
        
        # Exit when CMF crosses zero (change in money flow direction)
        exit_long = cmf_aligned[i] < 0
        exit_short = cmf_aligned[i] > 0
        
        # Execute trades
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals