#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d KAMA trend and RSI mean reversion.
# Uses KAMA (Kaufman Adaptive Moving Average) on 1d to determine trend direction,
# with RSI(14) on 4h for mean-reversion entries. Only takes long when KAMA is rising
# and RSI < 30, or short when KAMA is falling and RSI > 70.
# Volume confirmation ensures momentum. Designed to work in both bull and bear
# markets by adapting to trend conditions while avoiding overtrading.
# Target: 50-150 total trades over 4 years (12-37/year).
name = "4h_1d_KAMA_RSI_MeanRev"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA calculation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate KAMA on 1d timeframe
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(np.concatenate([[close_1d[0]], close_1d])))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder for correct calculation
    # Correct ER calculation: |close - close[10]| / sum(|close[i] - close[i-1]|) for i=1..10
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        direction = np.abs(close_1d[i] - close_1d[i-10])
        volatility = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
        if volatility > 0:
            er[i] = direction / volatility
        else:
            er[i] = 0
    # Smoothing constants
    sc = (er * 0.6 + 0.064) ** 2  # where 0.6 = 2/(2+1), 0.064 = 2/(30+1)
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI on 4h timeframe (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
            
        # Determine KAMA trend: rising if current > previous, falling if current < previous
        kama_rising = kama_aligned[i] > kama_aligned[i-1]
        kama_falling = kama_aligned[i] < kama_aligned[i-1]
        
        if position == 0:
            # Long when KAMA rising and RSI oversold with volume
            if kama_rising and rsi[i] < 30 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short when KAMA falling and RSI overbought with volume
            elif kama_falling and rsi[i] > 70 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when KAMA turns down or RSI overbought
            if not kama_rising or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when KAMA turns up or RSI oversold
            if not kama_falling or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals