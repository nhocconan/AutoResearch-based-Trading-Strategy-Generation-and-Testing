#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend with daily RSI filter and volume confirmation.
# Long when KAMA indicates uptrend AND daily RSI > 50 (bullish bias) AND volume > 1.5x 20-period average.
# Short when KAMA indicates downtrend AND daily RSI < 50 (bearish bias) AND volume > 1.5x 20-period average.
# Uses adaptive smoothing to reduce whipsaw in choppy markets.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency to avoid fee drag.

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
    
    # Daily data for RSI filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 14:
        return np.zeros(n)
    
    # KAMA calculation on 12h close
    er_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if len(change) < er_period else None
    if volatility is None:
        # Efficient calculation
        volatility = np.zeros_like(close)
        for i in range(er_period, len(close)):
            volatility[i] = volatility[i-1] - np.abs(close[i-er_period] - close[i-er_period-1]) + np.abs(close[i] - close[i-1])
        volatility[:er_period] = np.sum(np.abs(np.diff(close[:er_period], prepend=close[0])))
    
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA trend: 1 if close > kama, -1 if close < kama
    kama_trend = np.where(close > kama, 1, np.where(close < kama, -1, 0))
    
    # Daily RSI
    rsi_period = 14
    delta = np.diff(df_d['close'].values, prepend=df_d['close'].values[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align daily RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_d, rsi)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, er_period, rsi_period)  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_trend[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA uptrend, RSI > 50, volume filter
            long_cond = (kama_trend[i] == 1) and (rsi_aligned[i] > 50) and volume_filter[i]
            # Short conditions: KAMA downtrend, RSI < 50, volume filter
            short_cond = (kama_trend[i] == -1) and (rsi_aligned[i] < 50) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA turns down
            if kama_trend[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA turns up
            if kama_trend[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals