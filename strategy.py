#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1w Elder Ray Power + Volume Spike Confirmation
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w Elder Ray Power (bull/bear power from weekly candles) for regime filter.
- Williams %R(14): Extreme oversold (< -80) for long, overbought (> -20) for short.
- Volume confirmation: Current volume > 1.5 * 20-period volume MA.
- Exit: Reverse signal on opposite extreme or volume drying up.
- Signal size: 0.25 discrete to manage drawdown in volatile 6h bars.
- Works in bull/bear: Elder Ray regime filters counter-trend extremes; Williams %R captures mean reversion in extended moves.
- Novelty: Combines momentum extreme (Williams %R) with weekly power balance (Elder Ray) and volume confirmation - not recently tried on 6h.
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
    
    # Get 1w data for Elder Ray Power (HTF regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    ema_13 = pd.Series(df_1w_close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1w_high - ema_13  # Bull Power: High - EMA
    bear_power = df_1w_low - ema_13   # Bear Power: Low - EMA
    # Net Power: Bull Power + Bear Power (positive = bullish regime, negative = bearish regime)
    elder_ray_power = bull_power + bear_power
    
    # Get 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    df_6h_high = df_6h['high'].values
    df_6h_low = df_6h['low'].values
    df_6h_close = df_6h['close'].values
    
    highest_high = pd.Series(df_6h_high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_6h_low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_6h_close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align HTF indicators to 6h
    elder_ray_aligned = align_htf_to_ltf(prices, df_1w, elder_ray_power)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Calculate volume MA(20) for confirmation (using 6h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 13, 14, 20)  # Need enough bars for EMA13, Williams %R, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(elder_ray_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Long: Williams %R extremely oversold (< -80) AND Elder Ray bullish (> 0) AND volume confirmed
            if williams_r_aligned[i] < -80.0 and elder_ray_aligned[i] > 0 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R extremely overbought (> -20) AND Elder Ray bearish (< 0) AND volume confirmed
            elif williams_r_aligned[i] > -20.0 and elder_ray_aligned[i] < 0 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when Williams %R rises above -50 (momentum fading) OR volume dries up
            if williams_r_aligned[i] > -50.0 or curr_volume < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Williams %R falls below -50 (momentum fading) OR volume dries up
            if williams_r_aligned[i] < -50.0 or curr_volume < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1wElderRay_Power_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0