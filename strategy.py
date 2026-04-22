#!/usr/bin/env python3

"""
Hypothesis: 1-hour RSI(14) with 4-hour trend filter and volume confirmation.
Long when RSI < 30 (oversold), 4h EMA(20) trending up, and volume > 1.5x 20-period average.
Short when RSI > 70 (overbought), 4h EMA(20) trending down, and volume > 1.5x 20-period average.
Uses RSI for mean reversion entries aligned with 4h trend direction to avoid counter-trend trades.
Designed for low trade frequency (15-37/year) by requiring RSI extremes, trend alignment, and volume confirmation.
Works in both bull and bear markets by following the 4h trend direction.
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
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 4h data for trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA(20) for trend direction
    ema20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold), 4h EMA trending up, volume confirmation
            if rsi[i] < 30 and ema20_4h_aligned[i] > ema20_4h_aligned[i-1] and vol_confirmed:
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought), 4h EMA trending down, volume confirmation
            elif rsi[i] > 70 and ema20_4h_aligned[i] < ema20_4h_aligned[i-1] and vol_confirmed:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60) or opposite RSI extreme
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI >= 40 or RSI > 70 (overbought reversal)
                if rsi[i] >= 40 or rsi[i] > 70:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI <= 60 or RSI < 30 (oversold reversal)
                if rsi[i] <= 60 or rsi[i] < 30:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_RSI14_4hEMA20_Trend_Volume"
timeframe = "1h"
leverage = 1.0