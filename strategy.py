#!/usr/bin/env python3

"""
Hypothesis: 1-hour RSI(14) mean reversion with 4-hour trend filter and volume confirmation.
Enters long when RSI < 30 and price above 4h EMA50 (uptrend), short when RSI > 70 and price below 4h EMA50 (downtrend).
Exits when RSI returns to neutral (40-60) or trend reverses.
Uses 4h EMA50 for trend direction and 1h volume spike (>1.5x 20-period average) for confirmation.
Targets 15-30 trades/year (60-120 total over 4 years) to avoid fee drag.
Works in both bull and mean-reverting markets by combining trend alignment with RSI extremes.
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
    
    # Load 4h data for trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA for trend filter (50-period)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # RSI(14) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: RSI oversold (<30) and price above 4h EMA50 (uptrend)
            if rsi[i] < 30 and close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought (>70) and price below 4h EMA50 (downtrend)
            elif rsi[i] > 70 and close[i] < ema_50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: RSI returns to neutral (40-60) or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI >= 40 or price below 4h EMA50
                if rsi[i] >= 40 or close[i] < ema_50_4h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI <= 60 or price above 4h EMA50
                if rsi[i] <= 60 or close[i] > ema_50_4h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_RSI14_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0