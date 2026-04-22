# 12h_KAMA_1dRSI_VolumeBreakout - KAMA trend + RSI momentum + volume surge
# KAMA adapts to market noise - slow in chop, fast in trends. Combined with RSI > 50 for bullish bias
# and volume > 2x 20-period average for confirmation. Designed for 12h timeframe to target 12-37 trades/year.
# Works in bull markets by following KAMA trend, works in bear markets by avoiding false signals in chop.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Load 1d data for KAMA calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d close
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close_1d, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d, k=1)), axis=0)  # 10-period volatility
    er = np.where(volatility != 0, change / volatility, 0)
    sc = np.power(er * (2/2 - 2/30) + 2/30, 2)  # using fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate 14-period RSI on 1d close
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad beginning with NaN
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Align 1d indicators to 12h timeframe (waits for 1d bar to close)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price > KAMA + RSI > 50 + volume spike
            if price > kama_val and rsi_val > 50 and vol_spike:
                signals[i] = 0.25
                position = 1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price < KAMA or RSI < 40
                if price < kama_val or rsi_val < 40:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else 0.0
    
    return signals

name = "12h_KAMA_1dRSI_VolumeBreakout"
timeframe = "12h"
leverage = 1.0