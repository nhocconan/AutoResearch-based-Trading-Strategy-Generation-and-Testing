#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for EMA trend direction.
- EMA34 > EMA89 on 1d indicates bullish trend, EMA34 < EMA89 indicates bearish trend.
- In bullish trend: Long when price breaks above Camarilla R3 level with volume confirmation.
- In bearish trend: Short when price breaks below Camarilla S3 level with volume confirmation.
- Camarilla levels calculated from previous 1d OHLC: 
  R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (6h).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and EMAs
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 90:  # Need enough for EMA89
        return np.zeros(n)
    
    # Calculate 1d EMAs for trend filter
    ema_fast = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_slow = pd.Series(df_1d['close']).ewm(span=89, adjust=False, min_periods=89).mean().values
    trend_bullish = ema_fast > ema_slow  # True for bullish trend
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla levels use previous day's OHLC
    prev_close = pd.Series(df_1d['close']).shift(1).values
    prev_high = pd.Series(df_1d['high']).shift(1).values
    prev_low = pd.Series(df_1d['low']).shift(1).values
    
    # Calculate the range
    rang = prev_high - prev_low
    
    # Camarilla levels
    R4 = prev_close + 1.5 * rang
    R3 = prev_close + 1.1 * rang
    R2 = prev_close + 0.55 * rang
    R1 = prev_close + 0.275 * rang
    S1 = prev_close - 0.275 * rang
    S2 = prev_close - 0.55 * rang
    S3 = prev_close - 1.1 * rang
    S4 = prev_close - 1.5 * rang
    
    # Align 1d indicators to 6h
    ema_fast_aligned = align_htf_to_ltf(prices, df_1d, ema_fast)
    ema_slow_aligned = align_htf_to_ltf(prices, df_1d, ema_slow)
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish.astype(float))
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(90, 20)  # Need enough 1d bars for EMA89 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_fast_aligned[i]) or np.isnan(ema_slow_aligned[i]) or 
            np.isnan(trend_bullish_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        is_bullish_trend = trend_bullish_aligned[i] > 0.5
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if is_bullish_trend:
                    # Bullish trend: Long breakout above R3
                    if curr_high > R3_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                else:
                    # Bearish trend: Short breakdown below S3
                    if curr_low < S3_aligned[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price closes below R1 OR trend turns bearish
            if curr_close < R1_aligned[i] or is_bullish_trend == False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above S1 OR trend turns bullish
            if curr_close > S1_aligned[i] or is_bullish_trend == True:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_1dEMA34_89Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0