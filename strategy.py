#!/usr/bin/env python3
name = "4h_KAMA_Trend_RSI_Range"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA for trend direction (ER=10, FA=2, SC=30)
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = 0
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    volatility = np.concatenate([[0], volatility])
    for i in range(10, len(volatility)):
        volatility[i] = np.sum(np.abs(close[i-9:i+1] - close[i-10:i]))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI for range condition (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.nan_to_num(rsi, nan=50)
    
    # Volume surge filter (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price above KAMA (uptrend) AND RSI in oversold range (30-40) with volume
            if (close[i] > kama[i] and 
                30 <= rsi[i] <= 40 and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short entry: price below KAMA (downtrend) AND RSI in overbought range (60-70) with volume
            elif (close[i] < kama[i] and 
                  60 <= rsi[i] <= 70 and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses back below/above KAMA (trend reversal)
            if position == 1:
                if close[i] <= kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if close[i] >= kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals