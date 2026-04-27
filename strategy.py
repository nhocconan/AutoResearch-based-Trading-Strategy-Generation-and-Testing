#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour strategy using KAMA (Kaufman Adaptive Moving Average) as trend filter
# and RSI for mean-reversion entries. KAMA adapts to market noise, reducing whipsaw in choppy markets.
# RSI < 30 triggers long entries in uptrend; RSI > 70 triggers short entries in downtrend.
# Volume confirmation (>1.3x 20-period average) ensures institutional participation.
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (KAMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # KAMA (Kaufman Adaptive Moving Average) on daily close
    # ER (Efficiency Ratio) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(close_1d - np.roll(close_1d, 10))
    change[0:10] = np.nan  # Not enough data for first 10 periods
    abs_diff = np.abs(np.diff(close_1d, prepend=np.nan))
    volatility = pd.Series(abs_diff).rolling(window=10, min_periods=10).sum().values
    er = change / volatility
    er = np.where(volatility == 0, 0, er)  # Avoid division by zero
    
    # Smoothing constants: fastest SC = 2/(2+1) = 0.67, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 12h timeframe (wait for 1d bar to close)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI (14) on 12h close
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)  # Avoid division by zero
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: RSI < 30 (oversold) with price above KAMA (uptrend) and volume
        if (rsi[i] < 30 and 
            close[i] > kama_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short entry: RSI > 70 (overbought) with price below KAMA (downtrend) and volume
        elif (rsi[i] > 70 and 
              close[i] < kama_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: RSI returns to neutral zone (40-60)
        elif position == 1 and rsi[i] >= 40:
            signals[i] = 0.0
            position = 0
        elif position == -1 and rsi[i] <= 60:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_KAMA_RSI_VolumeFilter"
timeframe = "12h"
leverage = 1.0