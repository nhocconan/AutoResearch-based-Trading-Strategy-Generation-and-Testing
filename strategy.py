#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend with 1-day RSI filter and volume confirmation.
# Long when: KAMA direction up, RSI(1d) > 50, volume > 1.3x 20-period average
# Short when: KAMA direction down, RSI(1d) < 50, volume > 1.3x 20-period average
# Exit when: KAMA reverses direction or volume drops below average
# Designed for ~20-30 trades/year per symbol. Works in trending markets by filtering with higher timeframe momentum.
name = "12h_KAMA_RSI_Volume"
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 12h data
    def kama(price, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(price, prepend=price[0]))
        volatility = np.abs(np.diff(price, prepend=price[0]))
        for i in range(1, len(volatility)):
            volatility[i] = volatility[i-1] + np.abs(price[i] - price[i-1])
        
        er = np.zeros_like(price)
        for i in range(er_length, len(price)):
            if volatility[i] != 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama_val = np.zeros_like(price)
        kama_val[0] = price[0]
        for i in range(1, len(price)):
            kama_val[i] = kama_val[i-1] + sc[i] * (price[i] - kama_val[i-1])
        return kama_val
    
    kama_val = kama(close)
    kama_dir = np.where(kama_val > np.roll(kama_val, 1), 1, -1)
    kama_dir[0] = 1  # Initialize
    
    # Calculate RSI on daily data
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # Initial average
        avg_gain[length] = np.mean(gain[1:length+1])
        avg_loss[length] = np.mean(loss[1:length+1])
        
        # Wilder's smoothing
        for i in range(length+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val
    
    rsi_1d = rsi(close_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume average (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_dir[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_direction = kama_dir[i]
        rsi_val = rsi_1d_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: KAMA up, RSI > 50, volume confirmation
            if kama_direction == 1 and rsi_val > 50 and vol > 1.3 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: KAMA down, RSI < 50, volume confirmation
            elif kama_direction == -1 and rsi_val < 50 and vol > 1.3 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA reverses down or volume drops below average
            if kama_direction == -1 or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA reverses up or volume drops below average
            if kama_direction == 1 or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals