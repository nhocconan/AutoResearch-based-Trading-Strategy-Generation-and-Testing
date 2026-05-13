#!/usr/bin/env python3
# 12h_KAMA_Trend_Filter_Volume_Squeeze
# Hypothesis: Use KAMA direction as a noise-resistant trend filter on daily timeframe. Enter long/short when price breaks Bollinger Bands during low volatility (squeeze) in the direction of daily KAMA trend, confirmed by volume spike. Exit on Bollinger mean reversion or trend change. Designed to work in both bull and bear by capturing volatility breakouts in trending markets while avoiding whipsaws in chop.

name = "12h_KAMA_Trend_Filter_Volume_Squeeze"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data
    df_1d = get_htf_data(prices, '1d')
    
    # KAMA parameters
    er_length = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(df_1d['close'], prepend=df_1d['close'][0]))
    volatility = np.abs(np.diff(df_1d['close'], prepend=df_1d['close'][0]))
    for i in range(1, len(volatility)):
        volatility[i] = volatility[i-1] + np.abs(df_1d['close'][i] - df_1d['close'][i-1])
    
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # KAMA calculation
    kama = np.full_like(df_1d['close'], np.nan, dtype=float)
    kama[0] = df_1d['close'][0]
    for i in range(1, len(kama)):
        kama[i] = kama[i-1] + sc[i] * (df_1d['close'][i] - kama[i-1])
    
    # Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    sma20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    
    # Bollinger Band Width for squeeze detection
    bb_width = (upper - lower) / sma20
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Align daily indicators to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Volume spike: volume > 2.0 * 4-period average (1 day worth at 12h)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > 2.0 * vol_ma_4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(squeeze_aligned[i]) or 
            np.isnan(kama_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > upper band + squeeze + daily KAMA uptrend + volume spike
            if close[i] > upper_aligned[i] and squeeze_aligned[i] and close[i] > kama_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < lower band + squeeze + daily KAMA downtrend + volume spike
            elif close[i] < lower_aligned[i] and squeeze_aligned[i] and close[i] < kama_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below middle band (SMA20) OR price below KAMA
            sma20_aligned = align_htf_to_ltf(prices, df_1d, sma20)
            if close[i] < sma20_aligned[i] or close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above middle band (SMA20) OR price above KAMA
            sma20_aligned = align_htf_to_ltf(prices, df_1d, sma20)
            if close[i] > sma20_aligned[i] or close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals