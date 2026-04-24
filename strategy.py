#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1d Elder Ray Power + Volume Spike Confirmation
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d Elder Ray Power (Bull Power = High - EMA13, Bear Power = EMA13 - Low) for trend strength.
- Williams %R(14) on 6h for overbought/oversold extremes.
- Entry: Long when Williams %R < -80 (oversold) AND 1d Bull Power > 0 AND volume > 2.0 * volume MA(20).
         Short when Williams %R > -20 (overbought) AND 1d Bear Power > 0 AND volume > 2.0 * volume MA(20).
- Exit: Close-based reversal - exit long when Williams %R > -50, exit short when Williams %R < -50.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
Designed to work in both bull and bear markets via Elder Ray trend filter and mean-reversion exits.
Williams %R extremes capture exhaustion moves, Elder Ray confirms underlying trend strength.
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
    
    # Get 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R(14) on 6h
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_6h) / (highest_high - lowest_low) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 6h timeframe (no additional delay needed as it's contemporaneous)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Get 1d data for Elder Ray Power calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray Power: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13
    bear_power = ema_13 - low_1d
    
    # Align Elder Ray components to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate volume MA(20) for confirmation (using 6h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 14, 13, 20)  # Need enough bars for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_williams_r = williams_r_aligned[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Long: Williams %R < -80 (oversold) AND Bull Power > 0 (uptrend strength) AND volume confirmed
            if curr_williams_r < -80 and bull_power_aligned[i] > 0 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND Bear Power > 0 (downtrend strength) AND volume confirmed
            elif curr_williams_r > -20 and bear_power_aligned[i] > 0 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when Williams %R > -50 (recovering from oversold)
            if curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Williams %R < -50 (declining from overbought)
            if curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dElderRay_Power_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0