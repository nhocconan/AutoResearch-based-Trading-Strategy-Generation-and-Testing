#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day KAMA direction with RSI filter and volume confirmation
# Long when 1-day KAMA is rising (bullish trend), RSI < 40 (pullback), and volume > 1.3x average
# Short when 1-day KAMA is falling (bearish trend), RSI > 60 (bounce), and volume > 1.3x average
# Uses daily KAMA for trend direction, RSI for mean-reversion entries within trend, volume for confirmation
# Designed to work in bull markets via pullbacks in uptrend and in bear markets via bounces in downtrend
# Target: 20-30 trades per year (80-120 over 4 years) with 0.25 position sizing

name = "12h_1dKAMA_RSI_Volume_v3"
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
    
    # Calculate 1-day KAMA ( Kaufman Adaptive Moving Average )
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Efficiency Ratio (ER) over 10 periods
    change = abs(df_1d['close'].diff(10))
    volatility = df_1d['close'].diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(df_1d))
    kama[0] = df_1d['close'].iloc[0]
    for i in range(1, len(df_1d)):
        kama[i] = kama[i-1] + sc.iloc[i] * (df_1d['close'].iloc[i] - kama[i-1])
    
    # KAMA direction: 1 if rising, -1 if falling, 0 if flat
    kama_diff = np.diff(kama, prepend=kama[0])
    kama_dir = np.where(kama_diff > 0, 1, np.where(kama_diff < 0, -1, 0))
    
    # Align KAMA direction to 12h timeframe
    kama_dir_aligned = align_htf_to_ltf(prices, df_1d, kama_dir)
    
    # RSI (14) on 1-day closes
    delta = df_1d['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # neutral when no data
    
    # Align RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi.values)
    
    # Volume confirmation: >1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kama_dir_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising (uptrend), RSI < 40 (pullback), volume confirmation
            if kama_dir_aligned[i] == 1 and rsi_aligned[i] < 40 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling (downtrend), RSI > 60 (bounce), volume confirmation
            elif kama_dir_aligned[i] == -1 and rsi_aligned[i] > 60 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA turns down OR RSI > 60 (overbought in uptrend)
            if kama_dir_aligned[i] == -1 or rsi_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA turns up OR RSI < 40 (oversold in downtrend)
            if kama_dir_aligned[i] == 1 or rsi_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals