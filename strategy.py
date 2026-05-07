#!/usr/bin/env python3
name = "1d_Supertrend_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0

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
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Supertrend parameters
    atr_period = 10
    atr_multiplier = 3.0
    
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate ATR using Wilder's smoothing (RMA)
    atr = np.zeros(n)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, n):
        atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Calculate basic upper and lower bands
    hl_avg = (high + low) / 2
    upper_band = hl_avg + atr_multiplier * atr
    lower_band = hl_avg - atr_multiplier * atr
    
    # Initialize Supertrend
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    # Set initial values
    supertrend[atr_period-1] = upper_band[atr_period-1]
    direction[atr_period-1] = 1
    
    # Calculate Supertrend
    for i in range(atr_period, n):
        if close[i] > supertrend[i-1]:
            direction[i] = 1
        elif close[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1 and direction[i-1] == -1:
            supertrend[i] = lower_band[i]
        elif direction[i] == -1 and direction[i-1] == 1:
            supertrend[i] = upper_band[i]
        elif direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Weekly EMA for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume filter: volume above 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(supertrend[i]) or np.isnan(direction[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: Supertrend uptrend, above Supertrend, weekly uptrend, volume confirmation
            if direction[i] == 1 and close[i] > supertrend[i] and ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: Supertrend downtrend, below Supertrend, weekly downtrend, volume confirmation
            elif direction[i] == -1 and close[i] < supertrend[i] and ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Supertrend turns down or weekly trend changes
            if direction[i] == -1 or ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Supertrend turns up or weekly trend changes
            if direction[i] == 1 or ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 1d Supertrend with weekly trend filter and volume confirmation
# - Supertrend (ATR=10, multiplier=3) identifies trend direction and dynamic support/resistance
# - Weekly EMA20 trend filter ensures alignment with higher timeframe trend
# - Volume confirmation (1.5x average) reduces false signals
# - Long when: Supertrend uptrend, price above Supertrend, weekly uptrend, volume spike
# - Short when: Supertrend downtrend, price below Supertrend, weekly downtrend, volume spike
# - Exit when Supertrend reverses or weekly trend changes
# - Works in both bull and bear markets by following the trend on multiple timeframes
# - Position size 0.25 targets ~20-50 trades/year to minimize fee drag
# - Supertrend provides clear trend signals with built-in volatility adaptation
# - Weekly filter reduces whipsaws vs single timeframe signals
# - Aims for 80-200 total trades over 4 years (20-50/year) to stay within limits
# - Combines proven trend following with multi-timeframe confirmation for robustness