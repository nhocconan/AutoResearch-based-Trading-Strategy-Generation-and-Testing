#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1d Elder Ray Power + volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) for trend strength.
- Williams %R(14): long when < -80 (oversold), short when > -20 (overbought).
- Volume confirmation: current volume > 1.5 * 20-period volume MA to filter weak signals.
- Signal size: 0.25 discrete to balance return and drawdown control.
- Designed to catch reversals in extended moves with trend strength filter.
- Works in both bull and bear markets by using Elder Ray to confirm underlying trend.
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
    
    # Get 1d data for Elder Ray Power and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    williams_r_6h_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Calculate 1d EMA(13) for Elder Ray
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Elder Ray Power
    bull_power = df_1d['high'].values - ema_13_1d  # High - EMA13
    bear_power = ema_13_1d - df_1d['low'].values   # EMA13 - Low
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 1d volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 14, 13, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_6h_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma_aligned[i]
            
            # Long: Williams %R < -80 (oversold) AND Bull Power > 0 (uptrend strength) AND volume confirmed
            if williams_r_6h_aligned[i] < -80 and bull_power_aligned[i] > 0 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND Bear Power > 0 (downtrend strength) AND volume confirmed
            elif williams_r_6h_aligned[i] > -20 and bear_power_aligned[i] > 0 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: exit when Williams %R > -50 (momentum fading) OR Bear Power > Bull Power (trend weakening)
            if williams_r_6h_aligned[i] > -50 or bear_power_aligned[i] > bull_power_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when Williams %R < -50 (momentum fading) OR Bull Power > Bear Power (trend weakening)
            if williams_r_6h_aligned[i] < -50 or bull_power_aligned[i] > bear_power_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dElderRay_Power_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0