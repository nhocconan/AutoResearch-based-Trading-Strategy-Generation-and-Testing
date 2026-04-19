#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend + RSI momentum + 1d volume confirmation
# - KAMA(10) trend direction for primary signal
# - RSI(14) momentum filter: only take KAMA signals when RSI confirms momentum
# - 1d volume spike: volume > 1.5x 20-period average for conviction
# - Designed for 12h timeframe to reduce trade frequency and avoid fee drag
# - Works in both bull and bear markets by following trend with momentum confirmation

name = "12h_KAMA_RSI_1dVolume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # KAMA calculation (ER=10, FAST=2, SLOW=30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # placeholder, will fix below
    
    # Proper ER calculation
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if i >= 10:
            price_change = np.abs(close[i] - close[i-10])
            price_volatility = np.sum(np.abs(np.diff(close[i-10:i+1])))
            if price_volatility > 0:
                er[i] = price_change / price_volatility
            else:
                er[i] = 1.0
    
    # Smooth ER
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 12h: 1d has 2x 12h bars
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 2.0)
        
        if position == 0:
            # Look for long entry: price > KAMA + RSI > 50 + volume
            if close[i] > kama[i] and rsi[i] > 50 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price < KAMA + RSI < 50 + volume
            elif close[i] < kama[i] and rsi[i] < 50 and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on trend reversal or RSI overextended
            if close[i] < kama[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on trend reversal or RSI oversold
            if close[i] > kama[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals