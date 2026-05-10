#!/usr/bin/env python3
# 12h_1d_Camarilla_R1_S1_Breakout_Trend_Volume
# Hypothesis: 12h breakout above/below daily Camarilla R1/S1 with volume confirmation and daily trend filter.
# Designed for low trade frequency (<30/year) to avoid fee drag. Works in bull/bear via daily trend filter.

name = "12h_1d_Camarilla_R1_S1_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Daily high, low, close for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 = close + 1.1*(high-low)/12
    # Camarilla S1 = close - 1.1*(high-low)/12
    r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align R1 and S1 to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_200_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from daily EMA200
        bullish_trend = close[i] > ema_200_aligned[i]
        bearish_trend = close[i] < ema_200_aligned[i]
        
        # Volume confirmation (2.0x average)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above R1 in bullish trend with volume surge
            if close[i] > r1_aligned[i] and bullish_trend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S1 in bearish trend with volume surge
            elif close[i] < s1_aligned[i] and bearish_trend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price falls below S1
                if close[i] < s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above R1
                if close[i] > r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals