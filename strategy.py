#!/usr/bin/env python3
"""
12h Candlestick Pattern + 1d ATR Trend + Volume Spike
Long when bullish candle forms with price above 1d ATR mean and volume spike
Short when bearish candle forms with price below 1d ATR mean and volume spike
Exit on opposite signal or when price crosses 1d ATR mean
Designed for low trade frequency (15-25/year) to minimize fee drag
Works in both bull (momentum) and bear (mean reversion via reversal patterns) markets
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for ATR trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    # Calculate 14-period ATR on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR calculation using Wilder's smoothing (equivalent to RMA)
    atr_1d = np.zeros_like(tr)
    atr_1d[0] = tr[0]
    for i in range(1, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # ATR mean (20-period) for trend filter
    atr_mean_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align ATR mean to 12h timeframe
    atr_mean_aligned = align_htf_to_ltf(prices, df_1d, atr_mean_1d)
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after lookback periods
        # Skip if data not ready
        if (np.isnan(atr_mean_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish candle: close > open AND close > high of previous candle
            bullish = close[i] > prices['open'].iloc[i] and close[i] > high[i-1]
            # Bearish candle: close < open AND close < low of previous candle
            bearish = close[i] < prices['open'].iloc[i] and close[i] < low[i-1]
            
            # Long: Bullish candle with price above ATR mean and volume spike
            if bullish and close[i] > atr_mean_aligned[i] and volume[i] > 2.0 * vol_avg_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish candle with price below ATR mean and volume spike
            elif bearish and close[i] < atr_mean_aligned[i] and volume[i] > 2.0 * vol_avg_20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: bearish candle OR price crosses below ATR mean
                bearish = close[i] < prices['open'].iloc[i] and close[i] < low[i-1]
                if bearish or close[i] < atr_mean_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: bullish candle OR price crosses above ATR mean
                bullish = close[i] > prices['open'].iloc[i] and close[i] > high[i-1]
                if bullish or close[i] > atr_mean_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Candle_ATRTrend_Volume"
timeframe = "12h"
leverage = 1.0
#%%