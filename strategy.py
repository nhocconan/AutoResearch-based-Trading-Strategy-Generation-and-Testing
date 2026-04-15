#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ATR(14) for volatility filter
    tr1 = np.maximum(df_1d['high'].values[1:] - df_1d['low'].values[1:], np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1]))
    tr2 = np.absolute(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr = np.concatenate([np.array([np.nan]), np.maximum(tr1, tr2)])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Bollinger Bands (20, 2) on close
    sma = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_threshold[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr_14_aligned[i])):
            continue
        
        # Volatility filter: require 1d ATR > 0.5 * 4h ATR (ensures sufficient volatility)
        tr_4h = np.maximum(high[1:i+1] - low[1:i+1], np.abs(high[1:i+1] - close[:-1]))
        tr_4h = np.concatenate([np.array([np.nan]), np.maximum(tr_4h, np.abs(low[1:i+1] - close[:-1]))])
        atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values[i]
        if np.isnan(atr_14_4h) or atr_14_4h < 0.5 * atr_14_aligned[i]:
            continue
        
        # Trend filter: price above/below 1d EMA50
        trend_up = close[i] > ema_50_aligned[i]
        trend_down = close[i] < ema_50_aligned[i]
        
        # Long: close breaks above upper band + volume confirmation + uptrend
        if close[i] > upper[i] and volume[i] > vol_threshold[i] and trend_up:
            signals[i] = 0.25
        
        # Short: close breaks below lower band + volume confirmation + downtrend
        elif close[i] < lower[i] and volume[i] > vol_threshold[i] and trend_down:
            signals[i] = -0.25
        
        # Exit: close crosses back inside bands (mean reversion)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < upper[i]) or
               (signals[i-1] == -0.25 and close[i] > lower[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Bollinger_Breakout_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0