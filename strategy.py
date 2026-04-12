#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_ema_cross_volume_confirmation_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) and EMA(200) on daily closes
    close_1d_series = pd.Series(close_1d)
    ema_50 = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align EMAs to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume filter: current volume > 20-period average (on 4h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # warmup for EMA calculation
        # Skip if not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # EMA trend conditions
        ema_bullish = ema_50_aligned[i] > ema_200_aligned[i]
        ema_bearish = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_ok[i]
        
        # Entry signals
        long_signal = ema_bullish and vol_ok
        short_signal = ema_bearish and vol_ok
        
        # Exit when EMA crosses in opposite direction
        exit_long = ema_bearish
        exit_short = ema_bullish
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals