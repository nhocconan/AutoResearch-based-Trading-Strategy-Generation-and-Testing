#!/usr/bin/env python3
# 4h_12h_Camarilla_R1_S1_Breakout_Volume_Volatility_v1
# Hypothesis: 4h breakout at Camarilla R1/S1 levels with volume spike and volatility filter (ATR(30) < 0.8 * ATR(100)).
# Uses 12h EMA50 as trend filter. Designed for 15-25 trades/year to avoid fee drag. Works in bull/bear via trend filter.

name = "4h_12h_Camarilla_R1_S1_Breakout_Volume_Volatility_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # 1d data for Camarilla levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 = close + 1.1*(high-low)/12
    # Camarilla S1 = close - 1.1*(high-low)/12
    r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align R1 and S1 to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volatility filter: ATR(30) < 0.8 * ATR(100) (low volatility regime)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr_30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_100 = pd.Series(tr).rolling(window=100, min_periods=100).mean().values
    low_volatility = atr_30 < (0.8 * atr_100)
    
    # Volume confirmation (4.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr_30[i]) or
            np.isnan(atr_100[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 12h EMA50
        bullish_trend = close[i] > ema_50_aligned[i]
        bearish_trend = close[i] < ema_50_aligned[i]
        
        # Volume confirmation (4.0x average)
        volume_surge = volume[i] > 4.0 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above R1 in bullish trend with volume surge and low volatility
            if close[i] > r1_aligned[i] and bullish_trend and volume_surge and low_volatility[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S1 in bearish trend with volume surge and low volatility
            elif close[i] < s1_aligned[i] and bearish_trend and volume_surge and low_volatility[i]:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: price crosses below 12h EMA50 (trend change)
                if close[i] < ema_50_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price crosses above 12h EMA50 (trend change)
                if close[i] > ema_50_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals