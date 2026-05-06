#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day KAMA trend with volume confirmation and session filter
# Long when 1-day KAMA is rising and price > KAMA with volume > 1.5x 20-period average during active session (08-20 UTC)
# Short when 1-day KAMA is falling and price < KAMA with volume > 1.5x 20-period average during active session
# Uses daily KAMA for trend direction, volume for confirmation, session filter to avoid low liquidity periods
# Designed to capture trending moves in both bull and bear markets while avoiding choppy periods
# Target: 20-30 trades per year (80-120 over 4 years) with 0.25 position sizing

name = "12h_1dKAMA_Trend_Volume_Session_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day KAMA ( Kaufman Adaptive Moving Average )
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Efficiency Ratio (ER) over 10 periods
    change = abs(df_1d['close'].diff(10))
    volatility = df_1d['close'].diff().abs().rolling(window=10).sum()
    er = change / volatility
    er.replace([np.inf, -np.inf], 0, inplace=True)
    er.fillna(0, inplace=True)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(len(df_1d))
    kama[0] = df_1d['close'].iloc[0]
    for i in range(1, len(df_1d)):
        kama[i] = kama[i-1] + sc.iloc[i] * (df_1d['close'].iloc[i] - kama[i-1])
    
    # KAMA trend direction: rising if current > previous
    kama_rising = kama > np.roll(kama, 1)
    kama_rising[0] = False
    kama_falling = kama < np.roll(kama, 1)
    kama_falling[0] = False
    
    # Align KAMA and trend to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    kama_rising_aligned = align_htf_to_ltf(prices, df_1d, kama_rising.astype(float))
    kama_falling_aligned = align_htf_to_ltf(prices, df_1d, kama_falling.astype(float))
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after KAMA warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kama_aligned[i]) or np.isnan(kama_rising_aligned[i]) or 
            np.isnan(kama_falling_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising and price > KAMA with volume confirmation
            if kama_rising_aligned[i] > 0.5 and close[i] > kama_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling and price < KAMA with volume confirmation
            elif kama_falling_aligned[i] > 0.5 and close[i] < kama_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA falling or price < KAMA
            if kama_falling_aligned[i] > 0.5 or close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA rising or price > KAMA
            if kama_rising_aligned[i] > 0.5 or close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals