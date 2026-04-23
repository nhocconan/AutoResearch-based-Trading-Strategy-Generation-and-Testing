#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe strategy using 1d Williams %R extremes with 1w EMA50 trend filter and volume confirmation (>1.3x average).
- Uses 1d Williams %R for mean reversion signals (oversold/overbought)
- 1w EMA50 ensures trading with the higher timeframe trend
- Volume confirmation filters low-momentum breakouts
- Session filter: 08-20 UTC to avoid low-liquidity periods
- Position size: 0.25 (discrete level to minimize fee churn)
- Target: 12-37 trades/year (50-150 over 4 years) to avoid fee drag
- Works in bull/bear via trend filter and mean reversion at extremes
"""

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
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Volume confirmation: > 1.3x 24-period average (balanced for 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # 1d Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 12h timeframe (use prior completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24)  # EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(williams_r_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Williams %R extreme signals (using prior completed 1d bar)
        oversold = williams_r_aligned[i-1] <= -80  # Oversold condition
        overbought = williams_r_aligned[i-1] >= -20  # Overbought condition
        
        if position == 0:
            # Long: Williams %R oversold AND price > 1w EMA50 AND volume confirmation AND in session
            if oversold and volume_confirm and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought AND price < 1w EMA50 AND volume confirmation AND in session
            elif overbought and volume_confirm and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R rises above -50 (mean reversion) OR price < 1w EMA50 (trend flip)
            if williams_r_aligned[i] > -50 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R falls below -50 (mean reversion) OR price > 1w EMA50 (trend flip)
            if williams_r_aligned[i] < -50 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Extreme_1wEMA50_VolumeSpike_Session"
timeframe = "12h"
leverage = 1.0