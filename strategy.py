#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend + RSI(2) mean reversion + volume spike
# Uses KAMA for adaptive trend detection on 1d, RSI(2) for short-term mean reversion entries
# Volume confirmation requires 1.5x average volume to filter weak breakouts
# Designed to work in both bull and bear markets by following KAMA trend direction
# Target: 7-25 trades/year (30-100 total over 4 years) on 1d timeframe
# Prioritizes BTC/ETH performance with SOL as secondary

name = "1d_KAMA_Trend_RSI2_Volume_Spike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for regime filter (choppiness)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # sum of absolute changes
    # Handle the array operations properly
    change_padded = np.concatenate([np.full(10, np.nan), change])
    volatility_padded = np.concatenate([np.full(1, np.nan), 
                                       np.convolve(np.abs(np.diff(close, n=1)), np.ones(10), 'valid')])
    # Simpler approach: calculate ER manually
    er = np.full(n, np.nan)
    for i in range(10, n):
        price_change = np.abs(close[i] - close[i-10])
        sum_abs_changes = np.sum(np.abs(np.diff(close[i-9:i+1])))
        if sum_abs_changes > 0:
            er[i] = price_change / sum_abs_changes
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start with close
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI(2)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    # Wilder's smoothing for RSI
    for i in range(2, n):
        if i == 2:
            avg_gain[i] = np.mean(gain[0:2])
            avg_loss[i] = np.mean(loss[0:2])
        else:
            avg_gain[i] = (avg_gain[i-1] * 1 + gain[i-1]) / 2
            avg_loss[i] = (avg_loss[i-1] * 1 + loss[i-1]) / 2
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 20-period EMA on volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # KAMA trend + RSI(2) mean reversion + volume spike
        # Long: Price above KAMA (uptrend) + RSI(2) < 10 (oversold) + volume spike
        # Short: Price below KAMA (downtrend) + RSI(2) > 90 (overbought) + volume spike
        if position == 0:
            if (close[i] > kama[i] and rsi[i] < 10 and volume_spike):
                signals[i] = 0.25
                position = 1
            elif (close[i] < kama[i] and rsi[i] > 90 and volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI(2) > 70 (overbought) OR price below KAMA (trend change)
            if rsi[i] > 70 or close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI(2) < 30 (oversold) OR price above KAMA (trend change)
            if rsi[i] < 30 or close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals