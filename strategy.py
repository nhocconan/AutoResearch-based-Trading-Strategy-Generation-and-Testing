#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R Reversal with 1w EMA50 trend filter and volume spike confirmation.
- Uses Williams %R(14) from 1d for mean reversion signals (long when oversold, short when overbought)
- 1w EMA50 as trend filter (long only above, short only below)
- Volume > 2.0x 20-period average for confirmation
- Position size: 0.25 discrete level to minimize fee churn
- Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
- Works in both bull/bear via trend filter + volatility-adjusted mean reversion
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14) from 1d data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100.0
    williams_r = np.where((highest_high - lowest_low) == 0, -50.0, williams_r)  # avoid division by zero
    
    # Align Williams %R to 1d timeframe (using completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA(50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)  # Volume MA, Williams %R, EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(williams_r_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Williams %R signals (long when oversold < -80, short when overbought > -20)
        oversold = williams_r_aligned[i] < -80.0
        overbought = williams_r_aligned[i] > -20.0
        
        if position == 0:
            # Long: 1d Williams %R oversold AND price above 1w EMA50 AND volume confirmation
            if oversold and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: 1d Williams %R overbought AND price below 1w EMA50 AND volume confirmation
            elif overbought and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 1d Williams %R crosses above -50 (mean reversion) OR price crosses below 1w EMA50
            if williams_r_aligned[i] > -50.0 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 1d Williams %R crosses below -50 (mean reversion) OR price crosses above 1w EMA50
            if williams_r_aligned[i] < -50.0 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_Reversal_1wEMA50_VolumeSpike_Confirm_v1"
timeframe = "1d"
leverage = 1.0