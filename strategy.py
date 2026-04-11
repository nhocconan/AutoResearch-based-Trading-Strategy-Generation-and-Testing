#!/usr/bin/env python3
# 6h_1d_supertrend_volume_v1
# Strategy: 6-hour Supertrend with 1-day trend filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Uses Supertrend (ATR-based trend) on 6h for entry timing, filtered by 1-day EMA50 trend direction.
# Volume spikes confirm momentum. Works in both bull and bear markets by aligning with higher timeframe trend
# while capturing trend continuations. Targets 50-150 trades over 4 years to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_supertrend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d OHLC for EMA50
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe (wait for daily close)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.zeros(n)
    dir = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    # Calculate Supertrend
    for i in range(atr_period, n):
        # Upper band
        if i == atr_period:
            supertrend[i] = upper_band[i]
            dir[i] = 1
        else:
            if close[i-1] > supertrend[i-1]:
                supertrend[i] = max(lower_band[i], supertrend[i-1])
            else:
                supertrend[i] = min(upper_band[i], supertrend[i-1])
            
            # Determine direction
            if close[i] > supertrend[i]:
                dir[i] = 1
            else:
                dir[i] = -1
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(supertrend[i]) or 
            np.isnan(dir[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Supertrend signals
        supertrend_uptrend = dir[i] == 1
        supertrend_downtrend = dir[i] == -1
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: Supertrend uptrend with volume in uptrend
        long_signal = supertrend_uptrend and vol_confirmed and uptrend_1d
        
        # Short: Supertrend downtrend with volume in downtrend
        short_signal = supertrend_downtrend and vol_confirmed and downtrend_1d
        
        # Exit when Supertrend direction changes
        exit_long = position == 1 and not supertrend_uptrend
        exit_short = position == -1 and not supertrend_downtrend
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals