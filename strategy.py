#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI(2) mean reversion + volume spike
# Long when: KAMA rising (bullish trend) AND RSI(2) < 10 (extreme oversold) AND volume > 2.0x average
# Short when: KAMA falling (bearish trend) AND RSI(2) > 90 (extreme overbought) AND volume > 2.0x average
# Uses discrete sizing (0.25) to minimize fee churn. KAMA adapts to market noise, effective in both bull and bear regimes.
# Timeframe: 1d (primary), HTF: 1w for EMA34 trend filter (optional, not used in this version for simplicity).

name = "1d_KAMA_RSI2_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[i] - close[i-1]| over 10 periods
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA 2
    slow_sc = 2 / (30 + 1)  # for EMA 30
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # start after 10 periods
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(2)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Use Wilder's smoothing (alpha = 1/period)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[1] = np.mean(gain[:2]) if len(gain) >= 2 else 0
    avg_loss[1] = np.mean(loss[:2]) if len(loss) >= 2 else 0
    for i in range(2, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2
        avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for alignment (RSI starts at index 1)
    rsi = np.concatenate([[np.nan], rsi])
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = np.zeros(n)
    vol_ma_20[19:] = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[19:]
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(10, 2, 20)  # warmup for KAMA(10), RSI(2), volume MA
    
    for i in range(start_idx, n):
        # Skip if any indicator not ready
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_kama = kama[i]
        curr_kama_prev = kama[i-1]
        curr_rsi = rsi[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Handle position exits
        if position == 1:  # Long position
            # Exit when RSI(2) > 50 (mean reversion complete) or KAMA turns down
            if (curr_rsi > 50 or
                curr_kama < curr_kama_prev):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when RSI(2) < 50 (mean reversion complete) or KAMA turns up
            if (curr_rsi < 50 or
                curr_kama > curr_kama_prev):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: KAMA rising AND RSI(2) < 10 AND volume confirm
            if (curr_kama > curr_kama_prev and
                curr_rsi < 10 and
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: KAMA falling AND RSI(2) > 90 AND volume confirm
            elif (curr_kama < curr_kama_prev and
                  curr_rsi > 90 and
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals