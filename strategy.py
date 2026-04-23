#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 1d Williams %R extremes with 1w EMA trend filter and volume confirmation.
Long when 1d Williams %R < -80 (oversold) AND price > 1w EMA50 AND volume > 1.5x 20-period average.
Short when 1d Williams %R > -20 (overbought) AND price < 1w EMA50 AND volume > 1.5x 20-period average.
Exit when price crosses 1w EMA50 or Williams %R returns to neutral (-50).
Uses discrete position sizing (0.25) to control drawdown and fee churn.
Designed for 6h timeframe to target 12-37 trades/year per symbol (50-150 total over 4 years).
Williams %R identifies exhaustion points; 1w EMA filter ensures alignment with higher timeframe trend.
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
    
    # Calculate 1d Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R calculation
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r[highest_high_14 == lowest_low_14] = -50  # avoid division by zero
    
    # Calculate 1w EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        williams_r_val = williams_r_aligned[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold AND price above 1w EMA50 AND volume spike
            if (williams_r_val < -80 and price > ema_50_1w_val and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought AND price below 1w EMA50 AND volume spike
            elif (williams_r_val > -20 and price < ema_50_1w_val and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Exit: Price crosses 1w EMA50 OR Williams %R returns to neutral (-50)
            if position == 1 and (price <= ema_50_1w_val or williams_r_val >= -50):
                exit_signal = True
            elif position == -1 and (price >= ema_50_1w_val or williams_r_val <= -50):
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Extremes_1wEMA50Filter_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0