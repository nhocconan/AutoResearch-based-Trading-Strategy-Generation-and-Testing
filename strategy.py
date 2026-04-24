#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme + 1d Elder Ray power filter + volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- Williams %R(14) identifies overbought/oversold extremes (long when %R < -80, short when %R > -20).
- 1d Elder Ray power confirms trend alignment: bullish when Bear Power < 0 AND Bull Power > 0,
  bearish when Bull Power < 0 AND Bear Power > 0 (using EMA13).
- Volume confirmation: current volume > 1.5 * 20-period volume MA to avoid low-volatility false signals.
- Exit: Williams %R returns to neutral zone (-50) to prevent whipsaw in ranging markets.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
Designed to work in both bull and bear markets via Elder Ray trend filter and mean-reversion exits.
Williams %R captures momentum extremes while Elder Ray ensures alignment with higher timeframe power balance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray power calculation (EMA13)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    bull_power = df_1d_high - ema_1d
    bear_power = df_1d_low - ema_1d
    
    # Align HTF indicators to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate Williams %R(14) on 6h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    # Avoid division by zero
    rr = highest_high - lowest_low
    rr[rr == 0] = 1e-10
    williams_r = -100 * (highest_high - close) / rr
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, lookback, 13, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Long: Williams %R oversold (< -80) AND 1d Elder Ray bullish (Bear Power < 0 AND Bull Power > 0) AND volume confirmed
            if williams_r[i] < -80.0 and bear_power_aligned[i] < 0 and bull_power_aligned[i] > 0 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND 1d Elder Ray bearish (Bull Power < 0 AND Bear Power > 0) AND volume confirmed
            elif williams_r[i] > -20.0 and bull_power_aligned[i] < 0 and bear_power_aligned[i] > 0 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when Williams %R returns to neutral zone (-50) to avoid whipsaw
            if williams_r[i] > -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Williams %R returns to neutral zone (-50)
            if williams_r[i] < -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dElderRay_Power_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0