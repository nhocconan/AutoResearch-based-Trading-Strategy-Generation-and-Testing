#!/usr/bin/env python3
"""
6h_rsi_pullback_1d_trend_volume_v1
Hypothesis: Uses 60-period RSI on 6h for momentum and 1d EMA200 for trend filter. Long when RSI crosses above 50 (bullish momentum) with price above 1d EMA200, short when RSI crosses below 50 (bearish momentum) with price below 1d EMA200. Volume confirmation filters low-activity signals. Designed to capture trend continuation with mean-reversion entries in both bull and bear markets by aligning with higher timeframe direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_rsi_pullback_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 60-period RSI on 6h
    rsi_period = 60
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    avg_gain = wilders_smoothing(gain, rsi_period)
    avg_loss = wilders_smoothing(loss, rsi_period)
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(rsi_period*2, n):
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI crosses below 50 or trend changes
            if rsi[i] < 50 or close[i] < ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses above 50 or trend changes
            if rsi[i] > 50 or close[i] > ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: RSI crosses above 50 (bullish momentum) with price above 1d EMA200 (uptrend)
            if rsi[i] > 50 and rsi[i-1] <= 50 and close[i] > ema_200_1d_aligned[i] and vol_confirmed:
                position = 1
                signals[i] = 0.25
            # Short: RSI crosses below 50 (bearish momentum) with price below 1d EMA200 (downtrend)
            elif rsi[i] < 50 and rsi[i-1] >= 50 and close[i] < ema_200_1d_aligned[i] and vol_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals