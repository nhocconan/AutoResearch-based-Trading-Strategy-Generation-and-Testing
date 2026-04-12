#!/usr/bin/env python3
"""
4h_12h_keltner_breakout_v1
Uses Keltner Channel breakout on 12h with momentum confirmation and volume filter.
Long when price closes above upper KC after bullish momentum, short when closes below lower KC after bearish momentum.
Exit when price crosses middle line or volatility expands.
Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drag.
Works in both trending and ranging markets by combining volatility-based breakouts with momentum confirmation.
"""

name = "4h_12h_keltner_breakout_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 12h data for Keltner Channel calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Keltner Channel (20, 2) on 12h
    kc_length = 20
    kc_mult = 2.0
    
    # Middle line (EMA)
    middle = pd.Series(close_12h).ewm(span=kc_length, adjust=False, min_periods=kc_length).mean().values
    
    # Average True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=kc_length, adjust=False, min_periods=kc_length).mean().values
    
    # Upper and lower bands
    upper = middle + kc_mult * atr
    lower = middle - kc_mult * atr
    
    # Momentum: RSI(14) on 12h
    delta = pd.Series(close_12h).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align Keltner Channels and RSI to 4h
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower)
    middle_aligned = align_htf_to_ltf(prices, df_12h, middle)
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    
    # Volume confirmation on 4h: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(middle_aligned[i]) or np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price closes above upper KC with bullish momentum (RSI > 50) and volume
        if close[i] > upper_aligned[i] and rsi_aligned[i] > 50 and vol_confirm[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short entry: price closes below lower KC with bearish momentum (RSI < 50) and volume
        elif close[i] < lower_aligned[i] and rsi_aligned[i] < 50 and vol_confirm[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions
        elif position == 1 and close[i] <= middle_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= middle_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals