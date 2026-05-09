#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h ADX Trend Strength + 4h/1d EMA Trend Filter + Volume Spike
# ADX > 25 indicates strong trend, works in both bull and bear markets.
# 4h EMA200 and 1d EMA50 filter ensures we trade with higher timeframe momentum.
# Volume spike confirms institutional participation and reduces false signals.
# Target: 15-37 trades/year (60-150 over 4 years) to avoid fee drag.
name = "1h_ADX_Trend_4h1dEMA_Filter_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ADX (14-period)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        else:
            plus_dm[i] = 0
            
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
        else:
            minus_dm[i] = 0
            
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    # Initial values
    atr[13] = np.mean(tr[1:14])
    plus_dm_sum = np.sum(plus_dm[1:14])
    minus_dm_sum = np.sum(minus_dm[1:14])
    
    for i in range(14, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        plus_dm_sum = plus_dm_sum - (plus_dm_sum / 14) + plus_dm[i]
        minus_dm_sum = minus_dm_sum - (minus_dm_sum / 14) + minus_dm[i]
        plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
        minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
    
    dx = np.zeros(n)
    adx = np.zeros(n)
    for i in range(14, n):
        if plus_di[i] + minus_di[i] != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            dx[i] = 0
    
    # Initial ADX value
    adx[27] = np.mean(dx[14:28])
    for i in range(28, n):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Get 4h and 1d data for trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA200 for trend filter
    ema200_4h = pd.Series(df_4h['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF EMAs to 1h
    ema200_4h_1h = align_htf_to_ltf(prices, df_4h, ema200_4h)
    ema50_1d_1h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for ADX calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(ema200_4h_1h[i]) or np.isnan(ema50_1d_1h[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.8 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.8
        
        # Trend condition: ADX > 25 indicates strong trend
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Long: Strong trend + price above both EMAs + volume spike
            if strong_trend and close[i] > ema200_4h_1h[i] and close[i] > ema50_1d_1h[i] and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: Strong trend + price below both EMAs + volume spike
            elif strong_trend and close[i] < ema200_4h_1h[i] and close[i] < ema50_1d_1h[i] and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Weak trend OR price below 4h EMA200
            if adx[i] < 20 or close[i] < ema200_4h_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Weak trend OR price above 4h EMA200
            if adx[i] < 20 or close[i] > ema200_4h_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals