#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend filter with RSI mean reversion and volume spike confirmation
# KAMA adapts to market noise, reducing whipsaw in ranging markets
# RSI(14) < 30 for long, > 70 for short provides mean reversion edge
# Volume spike (>1.8x 20-period average) confirms institutional participation
# Designed for 4h timeframe targeting 25-40 trades/year with strong performance in both bull and bear markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for KAMA trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily closes
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    er = np.zeros_like(close_1d)
    sc = np.zeros_like(close_1d)
    kama = np.zeros_like(close_1d)
    
    # Calculate efficiency ratio over 10 periods
    for i in range(10, len(close_1d)):
        direction = np.abs(close_1d[i] - close_1d[i-10])
        volatility_sum = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
        if volatility_sum > 0:
            er[i] = direction / volatility_sum
        else:
            er[i] = 0
        sc[i] = (er[i] * (0.6667 - 0.0645) + 0.0645) ** 2
        if i == 10:
            kama[i] = close_1d[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI(14) for mean reversion
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price below KAMA (dip in uptrend) + RSI oversold + volume spike
            if (close[i] < kama_aligned[i] and 
                rsi[i] < 30 and 
                volume[i] > 1.8 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price above KAMA (rally in downtrend) + RSI overbought + volume spike
            elif (close[i] > kama_aligned[i] and 
                  rsi[i] > 70 and 
                  volume[i] > 1.8 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to KAMA or RSI reaches neutral zone
            if position == 1:
                # Exit long: price returns above KAMA or RSI > 50
                if (close[i] > kama_aligned[i] or 
                    rsi[i] > 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price returns below KAMA or RSI < 50
                if (close[i] < kama_aligned[i] or 
                    rsi[i] < 50):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_KAMA_RSI_MeanReversion_VolumeSpike"
timeframe = "4h"
leverage = 1.0