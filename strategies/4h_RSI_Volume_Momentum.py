# USDCAD? No, BTC. BTC is king. Let's focus on BTC and ETH.
# Hypothesis: A simple 4-hour momentum strategy using 4-hour RSI and volume spike.
# In both bull and bear markets, strong momentum bursts often precede continuation.
# We'll use RSI to detect momentum extremes and volume to confirm the strength of the move.
# Entry: RSI > 60 and volume > 1.5x 20-period average volume -> long
# Entry: RSI < 40 and volume > 1.5x 20-period average volume -> short
# Exit: RSI crosses back to neutral (50 for long, 50 for short) or opposite extreme.
# Position size: 0.25
# This should yield moderate trade frequency and avoid overtrading.

#!/usr/bin/env python3
import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate RSI on 4h close
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    alpha = 1.0 / 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:13] = np.nan  # Not enough data
    
    # Volume average
    volume = prices['volume'].values
    vol_ma = np.zeros_like(volume)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_ma[:20] = np.nan
    
    # Signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN
        if np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi[i]
        vol_val = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: RSI > 60 and volume spike
            if rsi_val > 60 and vol_val > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: RSI < 40 and volume spike
            elif rsi_val < 40 and vol_val > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI drops below 50
            if rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI rises above 50
            if rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Volume_Momentum"
timeframe = "4h"
leverage = 1.0