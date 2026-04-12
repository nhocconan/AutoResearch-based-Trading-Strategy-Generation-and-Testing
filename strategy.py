#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_ema_cross_volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA and volume calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 21 and 55 EMA on daily close
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_55_1d = pd.Series(close_1d).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Calculate 20-period volume moving average on daily data
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align to 12h timeframe
    ema_21_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    ema_55_aligned = align_htf_to_ltf(prices, df_1d, ema_55_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(ema_21_aligned[i]) or np.isnan(ema_55_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # EMA cross signals with volume confirmation
        # Golden Cross: EMA21 crosses above EMA55
        golden_cross = ema_21_aligned[i] > ema_55_aligned[i] and ema_21_aligned[i-1] <= ema_55_aligned[i-1]
        # Death Cross: EMA21 crosses below EMA55
        death_cross = ema_21_aligned[i] < ema_55_aligned[i] and ema_21_aligned[i-1] >= ema_55_aligned[i-1]
        
        # Volume filter: current 12h volume > 20-day average volume
        volume_ok = volume[i] > vol_ma_aligned[i]
        
        # Exit conditions
        exit_long = ema_21_aligned[i] < ema_55_aligned[i]
        exit_short = ema_21_aligned[i] > ema_55_aligned[i]
        
        # Execute trades
        if golden_cross and volume_ok and position != 1:
            position = 1
            signals[i] = 0.25
        elif death_cross and volume_ok and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals