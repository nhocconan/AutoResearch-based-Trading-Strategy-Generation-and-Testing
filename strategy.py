#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day KAMA trend with 1-week RSI filter and volume confirmation
# Long when KAMA(1d) is rising, RSI(1w) < 40 (oversold), and volume > 1.5x average
# Short when KAMA(1d) is falling, RSI(1w) > 60 (overbought), and volume > 1.5x average
# KAMA adapts to market noise, providing smooth trend signals. Weekly RSI identifies extremes.
# Volume confirms momentum. Works in both bull and bear markets by fading extremes in trends.
# Target: 25-40 trades per year (100-160 over 4 years) with 0.25 position sizing.

name = "4h_1dKAMA_1wRSI_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day KAMA trend ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Efficiency Ratio for KAMA
    change = abs(df_1d['close'] - df_1d['close'].shift(10))
    volatility = abs(df_1d['close'] - df_1d['close'].shift(1)).rolling(window=10).sum()
    er = change / volatility
    er = er.fillna(0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(df_1d))
    kama[0] = df_1d['close'].iloc[0]
    for i in range(1, len(df_1d)):
        kama[i] = kama[i-1] + sc.iloc[i] * (df_1d['close'].iloc[i] - kama[i-1])
    
    # KAMA trend: 1 if rising, -1 if falling
    kama_trend = np.where(kama > np.roll(kama, 1), 1, -1)
    kama_trend[0] = 0
    
    # Align KAMA trend to 4h timeframe
    kama_trend_aligned = align_htf_to_ltf(prices, df_1d, kama_trend)
    
    # Calculate 1-week RSI for overbought/oversold filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # RSI calculation
    delta = df_1w['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # neutral when no data
    
    # Align RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi.values)
    
    # Volume confirmation: >1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kama_trend_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising, RSI oversold (<40), volume confirmation
            if kama_trend_aligned[i] == 1 and rsi_aligned[i] < 40 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI overbought (>60), volume confirmation
            elif kama_trend_aligned[i] == -1 and rsi_aligned[i] > 60 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA turns falling or RSI overbought (>70)
            if kama_trend_aligned[i] == -1 or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA turns rising or RSI oversold (<30)
            if kama_trend_aligned[i] == 1 or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals