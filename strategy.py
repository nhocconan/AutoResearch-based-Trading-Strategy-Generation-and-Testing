#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Williams %R extremes with 4h EMA trend filter and volume confirmation.
Long when 1d Williams %R < -80 (oversold) AND price > 4h EMA(50) AND volume > 1.3x 20-period average.
Short when 1d Williams %R > -20 (overbought) AND price < 4h EMA(50) AND volume > 1.3x 20-period average.
Exit when price crosses 4h EMA(50) in opposite direction or ATR stoploss hit (2.5*ATR).
Uses discrete position sizing (0.25) to control drawdown and fee churn.
Designed for 4h timeframe to target 19-50 trades/year per symbol (75-200 total over 4 years).
Williams %R on 1d provides institutional overbought/oversold levels from higher timeframe.
EMA filter ensures we trade with the intermediate trend, reducing whipsaw.
Volume confirmation filters low-conviction breakouts.
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
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 4h timeframe (no extra delay needed for Williams %R)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 4h EMA(50) for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        wr_val = williams_r_aligned[i]
        ema_val = ema_50[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price above EMA(50) AND volume spike
            if (wr_val < -80 and price > ema_val and volume[i] > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Williams %R overbought (> -20) AND price below EMA(50) AND volume spike
            elif (wr_val > -20 and price < ema_val and volume[i] > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price crosses EMA(50) in opposite direction
            if position == 1 and price < ema_val:
                exit_signal = True
            elif position == -1 and price > ema_val:
                exit_signal = True
            
            # ATR-based stoploss: 2.5 * ATR from entry
            if position == 1 and price < entry_price - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_EMA50_VolumeConfirmation_ATRStop"
timeframe = "4h"
leverage = 1.0