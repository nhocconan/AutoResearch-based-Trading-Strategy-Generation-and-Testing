#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 1d Williams %R extremes with 1d EMA34 trend filter and volume confirmation.
Long when 1d Williams %R < -80 (oversold) AND price > 1d EMA34 (uptrend) AND 6h volume > 1.5x 20-period average.
Short when 1d Williams %R > -20 (overbought) AND price < 1d EMA34 (downtrend) AND 6h volume > 1.5x 20-period average.
Exit when price crosses 1d EMA34 in opposite direction or Williams %R returns to neutral range (-50 to -50).
Uses discrete position sizing (0.25) to control drawdown and fee churn.
Designed for 6h timeframe to target 12-37 trades/year per symbol (50-150 total over 4 years).
Williams %R identifies overextended moves; EMA34 filter ensures trades align with higher timeframe trend.
Works in both bull and bear markets by buying oversold dips in uptrends and selling overbought rallies in downtrends.
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
    
    # Calculate 1d Williams %R and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need at least 35 for EMA34 and 14 for Williams %R
        return np.zeros(n)
    
    # Williams %R(14) = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_1d['close'].values) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # EMA34
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(35, 20)  # EMA34 needs 35, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        williams_r_val = williams_r_aligned[i]
        ema_34_val = ema_34_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price > EMA34 (uptrend) AND volume spike
            if (williams_r_val < -80 and price > ema_34_val and volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price < EMA34 (downtrend) AND volume spike
            elif (williams_r_val > -20 and price < ema_34_val and volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price crosses EMA34 in opposite direction
            if position == 1 and price < ema_34_val:
                exit_signal = True
            elif position == -1 and price > ema_34_val:
                exit_signal = True
            
            # Secondary exit: Williams %R returns to neutral range (-50 to -50)
            if position == 1 and williams_r_val > -50:
                exit_signal = True
            elif position == -1 and williams_r_val < -50:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Extremes_EMA34Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0