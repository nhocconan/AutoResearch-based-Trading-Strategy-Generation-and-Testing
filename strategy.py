#!/usr/bin/env python3
"""
1h RSI(2) Extreme Reversion with 4h Trend Filter and Volume Spike
Long: RSI(2)<10 + price>4h EMA50 + volume spike
Short: RSI(2)>90 + price<4h EMA50 + volume spike
Exit: RSI(2) crosses above 50 (long) or below 50 (short)
Uses 4h EMA50 for trend filter, volume spike for entry confirmation.
Designed to capture mean reversion in both trending and ranging markets.
Target: 80-160 total trades over 4 years (20-40/year)
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
    
    # RSI(2) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[1] = gain[1]
    avg_loss[1] = loss[1]
    for i in range(2, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2
        avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need RSI and EMA calculations
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_50 = ema_50_aligned[i]
        
        if position == 0:
            # Long: RSI(2)<10 + price>4h EMA50 + volume spike
            if rsi[i] < 10 and price > ema_50 and vol_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: RSI(2)>90 + price<4h EMA50 + volume spike
            elif rsi[i] > 90 and price < ema_50 and vol_spike[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI(2) crosses above 50
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI(2) crosses below 50
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI2_Extreme_4hEMA50_VolumeSpike"
timeframe = "1h"
leverage = 1.0