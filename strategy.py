#!/usr/bin/env python3
# Hypothesis: 4h KAMA trend with RSI momentum filter and volume spike confirmation
# Long when: KAMA rising, RSI > 55, volume spike (>1.5x 20-period average)
# Short when: KAMA falling, RSI < 45, volume spike
# Exit when: KAMA reverses direction OR price crosses KAMA
# Position size: 0.25 to limit drawdown. Target: 20-50 trades/year.
# Designed to work in trending markets while avoiding whipsaws in ranging conditions.

name = "4h_KAMA_RSI_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # 10-period volatility
    # Fix array dimensions
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    
    # Calculate ER and smoothing constants
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # Fast=2, Slow=30
    sc = np.where(np.isnan(er), 0, sc)
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    # First average
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    # Wilder smoothing
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (1.5 * vol_ma)
    
    # Get 1d data for trend filter (more stable than 12h)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i < 50:
            ema_50_1d[i] = np.nan
        elif i == 50:
            ema_50_1d[i] = np.mean(close_1d[:51])
        else:
            ema_50_1d[i] = (close_1d[i] * 2/(50+1)) + (ema_50_1d[i-1] * (1 - 2/(50+1)))
    
    ema_50_1d_prev = np.roll(ema_50_1d, 1)
    ema_50_1d_prev[0] = ema_50_1d[0]
    ema_rising = ema_50_1d > ema_50_1d_prev
    ema_falling = ema_50_1d < ema_50_1d_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_spike[i]) or
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA rising, RSI > 55, volume spike
            if (kama[i] > kama[i-1] and 
                rsi[i] > 55 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling, RSI < 45, volume spike
            elif (kama[i] < kama[i-1] and 
                  rsi[i] < 45 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: KAMA turns down OR price crosses below KAMA
            if (kama[i] < kama[i-1]) or (close[i] < kama[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA turns up OR price crosses above KAMA
            if (kama[i] > kama[i-1]) or (close[i] > kama[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals