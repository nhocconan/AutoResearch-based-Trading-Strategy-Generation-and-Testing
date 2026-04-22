#!/usr/bin/env python3
"""
Hypothesis: 4-hour RSI(2) Extreme with 12-hour Trend and Volume Confirmation.
Long when RSI(2) < 10, 12h EMA50 rising, volume spike, price above SMA200.
Short when RSI(2) > 90, 12h EMA50 falling, volume spike, price below SMA200.
Exit when RSI(2) crosses above 50 (long) or below 50 (short).
Designed for low trade frequency by requiring extreme oversold/overbought conditions.
Works in both bull and bear markets by following the 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(2)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi2 = 100 - (100 / (1 + rs))
    
    # SMA200 for trend filter
    sma200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 12h close for trend
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(rsi2[i]) or np.isnan(sma200[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: RSI(2) < 10, 12h EMA50 rising, volume spike, price above SMA200
            if (rsi2[i] < 10 and ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and 
                vol_spike and close[i] > sma200[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI(2) > 90, 12h EMA50 falling, volume spike, price below SMA200
            elif (rsi2[i] > 90 and ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and 
                  vol_spike and close[i] < sma200[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI(2) crosses above 50 (long) or below 50 (short)
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI(2) crosses above 50
                if rsi2[i] >= 50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI(2) crosses below 50
                if rsi2[i] <= 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_RSI2_Extreme_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0