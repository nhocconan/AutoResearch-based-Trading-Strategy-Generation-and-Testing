#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 1d Williams %R extremes with 6h EMA50 trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND price > 6h EMA50 AND volume > 1.5x 20-period average.
Short when Williams %R > -20 (overbought) AND price < 6h EMA50 AND volume > 1.5x 20-period average.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
Uses discrete position sizing (0.25) to control drawdown and fee churn.
Designed for 6h timeframe to target 12-37 trades/year per symbol (50-150 total over 4 years).
Williams %R is a momentum oscillator that identifies overbought/oversold conditions, effective in both ranging and trending markets when combined with trend filter.
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
    
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          ((highest_high - close_1d) / (highest_high - lowest_low)) * -100, 
                          -50)
    
    # Align 1d Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 6h EMA50 for trend filter
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_6h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_6h, ema_50)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 50, 20)  # Williams %R needs 14, EMA needs 50, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        wr_val = williams_r_aligned[i]
        ema_50_val = ema_50_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price > 6h EMA50 AND volume spike
            if (wr_val < -80 and price > ema_50_val and volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price < 6h EMA50 AND volume spike
            elif (wr_val > -20 and price < ema_50_val and volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts)
            if position == 1 and wr_val > -50:
                exit_signal = True
            elif position == -1 and wr_val < -50:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Extremes_6hEMA50_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0