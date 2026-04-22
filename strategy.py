#!/usr/bin/env python3
"""
Hypothesis: 4h 10-period KAMA trend with 1d RSI momentum and volume confirmation.
Long when KAMA turns upward, RSI > 50 (bullish momentum), and volume > 1.5x average.
Short when KAMA turns downward, RSI < 50 (bearish momentum), and volume > 1.5x average.
Exit when KAMA reverses direction or RSI crosses 50.
Designed for low trade frequency (~25-35/year) to minimize fee drift.
Works in both bull and bear markets by using adaptive KAMA and momentum filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for RSI filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate KAMA (10-period) on 4h
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation
    close_series = pd.Series(close)
    change = abs(close_series.diff())
    volatility = abs(close_series.diff()).rolling(window=10, min_periods=1).sum()
    er = change / volatility.replace(0, np.nan)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate 1d RSI (14-period)
    close_d = pd.Series(df_daily['close'].values)
    delta = close_d.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_d = 100 - (100 / (1 + rs))
    
    # Align RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_daily, rsi_d.values)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):  # start after KAMA warmup
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA turning up, RSI > 50, volume spike
            if (kama[i] > kama[i-1] and 
                rsi_aligned[i] > 50 and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down, RSI < 50, volume spike
            elif (kama[i] < kama[i-1] and 
                  rsi_aligned[i] < 50 and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: KAMA turns down OR RSI < 50
                if kama[i] < kama[i-1] or rsi_aligned[i] < 50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: KAMA turns up OR RSI > 50
                if kama[i] > kama[i-1] or rsi_aligned[i] > 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_KAMA_1dRSI_Momentum_Volume"
timeframe = "4h"
leverage = 1.0
#%%