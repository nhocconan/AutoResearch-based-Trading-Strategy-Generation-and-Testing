#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend following with 1d RSI filter and volume confirmation.
# KAMA adapts to market efficiency - slow in ranging, fast in trending markets.
# Long when KAMA turns up AND price > KAMA AND 1d RSI < 70 (avoid overbought) AND volume > 1.5x average.
# Short when KAMA turns down AND price < KAMA AND 1d RSI > 30 (avoid oversold) AND volume > 1.5x average.
# Exit when price crosses KAMA in opposite direction.
# Uses adaptive trend following to capture trends while avoiding whipsaws in ranging markets.
# Target: 60-120 total trades over 4 years (15-30/year) for low fee drift.

name = "12h_KAMA_1dRSI_Volume"
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
    
    # 12h KAMA ( Kaufman Adaptive Moving Average )
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Proper ER calculation
    price_change = np.abs(close - np.roll(close, 10))  # 10-period change
    volatility_sum = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i-9)
        volatility_sum[i] = np.sum(np.abs(np.diff(close[start_idx:i+1]))) if i > 0 else 0
    
    er = np.where(volatility_sum != 0, price_change / volatility_sum, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: 1 if rising, -1 if falling
    kama_dir = np.where(kama > np.roll(kama, 1), 1, -1)
    kama_dir[0] = 0
    
    # 12h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate RSI (14-period) on 1d data
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[np.isnan(rsi)] = 50  # neutral when undefined
    
    # Align 1d RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(kama_dir[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA turning up, price > KAMA, RSI not overbought, volume spike
            long_cond = (kama_dir[i] == 1) and (close[i] > kama[i]) and (rsi_aligned[i] < 70) and volume_filter[i]
            # Short conditions: KAMA turning down, price < KAMA, RSI not oversold, volume spike
            short_cond = (kama_dir[i] == -1) and (close[i] < kama[i]) and (rsi_aligned[i] > 30) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals