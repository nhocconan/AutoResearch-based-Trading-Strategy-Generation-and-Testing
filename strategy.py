#!/usr/bin/env python3
"""
Hypothesis: Daily price action with weekly trend filter for multi-timeframe trend following.
Long when price > weekly EMA21 and daily RSI(14) > 50 with volume confirmation.
Short when price < weekly EMA21 and daily RSI(14) < 50 with volume confirmation.
Exit when price crosses weekly EMA21 or RSI crosses 50.
Designed for low trade frequency by requiring trend alignment and momentum confirmation.
Works in both bull and bear markets by following the weekly trend.
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
    
    # Daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # 21-period EMA on weekly close for trend
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema21_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price > weekly EMA21, RSI > 50, volume spike
            if (close[i] > ema21_1w_aligned[i] and rsi[i] > 50 and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price < weekly EMA21, RSI < 50, volume spike
            elif (close[i] < ema21_1w_aligned[i] and rsi[i] < 50 and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses weekly EMA21 or RSI crosses 50
            exit_signal = False
            
            if position == 1:
                # Exit long: Price <= weekly EMA21 or RSI <= 50
                if close[i] <= ema21_1w_aligned[i] or rsi[i] <= 50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price >= weekly EMA21 or RSI >= 50
                if close[i] >= ema21_1w_aligned[i] or rsi[i] >= 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_WeeklyTrend_RSI_Volume"
timeframe = "1d"
leverage = 1.0