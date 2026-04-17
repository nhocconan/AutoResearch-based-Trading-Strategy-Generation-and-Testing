#!/usr/bin/env python3
"""
4h_RSI_4_Trend_Filter_v1
RSI(4) extreme + 1d EMA50 trend filter + volume confirmation. 
Long when RSI(4)<15 + price>1d EMA50 + volume>1.5x avg. 
Short when RSI(4)>85 + price<1d EMA50 + volume>1.5x avg.
Exit when RSI returns to neutral (45-55) or trend reverses.
Designed for low-frequency, high-conviction entries in both bull and bear markets.
Target: 50-120 total trades over 4 years (12-30/year).
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
    
    # === RSI(4) for entry signal ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=4, adjust=False, min_periods=4).mean().values
    avg_loss = pd.Series(loss).ewm(span=4, adjust=False, min_periods=4).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume average for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d EMA50 for higher timeframe trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: RSI(4) oversold (<15), price above 1d EMA50, volume confirmed
            if (rsi[i] < 15 and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
                continue
            # Short: RSI(4) overbought (>85), price below 1d EMA50, volume confirmed
            elif (rsi[i] > 85 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: RSI returns to neutral or trend reverses
        elif position == 1:
            # Exit long: RSI returns to neutral (45-55) OR price crosses below 1d EMA50
            if (45 <= rsi[i] <= 55 or 
                close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (45-55) OR price crosses above 1d EMA50
            if (45 <= rsi[i] <= 55 or 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_4_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0