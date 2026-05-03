#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI(2) extreme and volume confirmation.
# Long when KAMA is rising, RSI(2) < 10 (deep pullback in uptrend), and volume > 1.5x 20-day MA.
# Short when KAMA is falling, RSI(2) > 90 (overbought bounce in downtrend), and volume > 1.5x 20-day MA.
# KAMA adapts to market noise, reducing whipsaws in ranging markets.
# RSI(2) captures short-term extremes for mean reversion within the trend.
# Volume confirmation ensures institutional participation.
# Designed for 1d timeframe to achieve 30-100 total trades over 4 years (7-25/year).

name = "1d_KAMA_RSI2_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # start at index 9 (10th element, 0-based)
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(2)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Use Wilder's smoothing (alpha = 1/period)
    avg_gain = np.full_like(close, np.nan, dtype=float)
    avg_loss = np.full_like(close, np.nan, dtype=float)
    avg_gain[1] = np.mean(gain[1:3])  # first average of first 2 gains
    avg_loss[1] = np.mean(loss[1:3])  # first average of first 2 losses
    for i in range(2, n):
        avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2
        avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection (20-period volume MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for stoploss
    
    for i in range(20, n):  # start from 20 to ensure indicators are valid
        # Skip if any value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        rsi_oversold = rsi[i] < 10
        rsi_overbought = rsi[i] > 90
        
        if position == 0:
            # Long: KAMA rising, RSI(2) < 10, volume spike
            if kama_rising and rsi_oversold and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: KAMA falling, RSI(2) > 90, volume spike
            elif kama_falling and rsi_overbought and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Exit: KAMA starts falling
            if not kama_rising:
                exit_signal = True
            # Exit: RSI(2) > 70 (overbought)
            elif rsi[i] > 70:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # Exit: KAMA starts rising
            if not kama_falling:
                exit_signal = True
            # Exit: RSI(2) < 30 (oversold)
            elif rsi[i] < 30:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals