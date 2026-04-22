# 1d_KAMA_Direction_RSI_Filter_v1
# Hypothesis: Daily trend following using KAMA for trend direction with RSI filter and volume confirmation.
# KAMA adapts to market noise - slow in ranging markets, fast in trending markets.
# Long when KAMA turns up, RSI > 50, and volume above average.
# Short when KAMA turns down, RSI < 50, and volume above average.
# Designed for low trade frequency (~8-15/year) with strong trend capture in both bull and bear markets.
# Uses daily timeframe to avoid noise and reduce whipsaw.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for KAMA and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) with ER=10, FC=2, SC=30
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # |close[t] - close[t-10]|
    volatility = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        volatility[i] = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
    
    # Avoid division by zero
    er = np.zeros_like(close_1d)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    
    # Smoothing Constants
    sc = 2  # fast SC
    ss = 30  # slow SC
    fc = er * (sc - ss) + ss  # smoothing constant
    fc = np.clip(fc, ss, sc)  # bound between slow and fast
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + fc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI(14) on daily
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = 50  # neutral before enough data
    
    # Align KAMA and RSI to daily timeframe (no alignment needed as we're using 1d data on 1d timeframe)
    kama_aligned = kama
    rsi_aligned = rsi
    
    # Calculate 20-day average volume for volume confirmation
    volume = df_1d['volume'].values
    vol_ma_20 = np.zeros_like(volume)
    for i in range(20, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after KAMA and RSI warmup
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_confirm = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: KAMA turning up, RSI > 50, volume confirmation
            if i > 0 and kama_val > kama_aligned[i-1] and rsi_val > 50 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down, RSI < 50, volume confirmation
            elif i > 0 and kama_val < kama_aligned[i-1] and rsi_val < 50 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: KAMA reverses direction
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when KAMA turns down
                if i > 0 and kama_val < kama_aligned[i-1]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when KAMA turns up
                if i > 0 and kama_val > kama_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_Direction_RSI_Filter_v1"
timeframe = "1d"
leverage = 1.0