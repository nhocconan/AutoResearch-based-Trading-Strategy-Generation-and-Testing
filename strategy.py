#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA with 1-day RSI filter and volume spike confirmation.
# Long when: KAMA rising, RSI(1d) < 30 (oversold), volume > 2x 20-period average
# Short when: KAMA falling, RSI(1d) > 70 (overbought), volume > 2x 20-period average
# Exit when: KAMA direction reverses or RSI returns to neutral (40-60)
# Designed for ~15-25 trades/year per symbol. Works in both bull and bear markets by fading extremes in ranging conditions and following KAMA in trending conditions.
name = "12h_KAMA_RSI_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate KAMA on 12h data
    def kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close))
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing Constant
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        # KAMA
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_val = kama(close, 10, 2, 30)
    kama_dir = np.diff(kama_val, prepend=kama_val[0])  # 1 if rising, -1 if falling
    
    # Calculate RSI on daily data
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=length, min_periods=length).mean().values
        avg_loss = pd.Series(loss).rolling(window=length, min_periods=length).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = rsi(close_1d, 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume spike: > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_dir[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        kama_direction = kama_dir[i]
        rsi_val = rsi_1d_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long: KAMA rising, RSI oversold, volume spike
            if kama_direction > 0 and rsi_val < 30 and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI overbought, volume spike
            elif kama_direction < 0 and rsi_val > 70 and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA falling or RSI returns to neutral
            if kama_direction < 0 or (rsi_val >= 40 and rsi_val <= 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA rising or RSI returns to neutral
            if kama_direction > 0 or (rsi_val >= 40 and rsi_val <= 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals